module Conv #(
    parameter INW = 24,
    parameter R = 9,
    parameter C = 8,
    parameter MAXK = 4,
    localparam OUTW = $clog2(MAXK*MAXK*(128'd1 << 2*INW-2) + (1 << (INW-1))) + 1,
    localparam K_BITS = $clog2(MAXK+1)
)(
    input clk, 
    input reset,
    input [INW-1:0] INPUT_TDATA,
    input INPUT_TVALID,
    input [K_BITS:0] INPUT_TUSER, 
    output INPUT_TREADY, 
    output [OUTW-1:0] OUTPUT_TDATA,
    output OUTPUT_TVALID,
    input OUTPUT_TREADY
);

    // Computed parameters for submodules
    localparam FIFO_DEPTH = C - 1;
    localparam X_ADDR_BITS = $clog2(R*C);
    localparam W_ADDR_BITS = $clog2(MAXK*MAXK);
    localparam R_BITS = $clog2(R);
    localparam C_BITS = $clog2(C);
    localparam K_CNT_BITS = $clog2(MAXK);
    
    // Signals from input_mems module
    logic inputs_loaded;
    logic compute_finished;
    logic [K_BITS-1:0] K;
    logic signed [INW-1:0] B;
    logic [X_ADDR_BITS-1:0] X_read_addr;
    logic signed [INW-1:0] X_data;
    logic [W_ADDR_BITS-1:0] W_read_addr;
    logic signed [INW-1:0] W_data;
    
    // MAC signals
    logic signed [INW-1:0] mac_input0, mac_input1, mac_init_value;
    logic signed [OUTW-1:0] mac_out;
    logic mac_init_acc, mac_input_valid;
    
    // FIFO signals
    logic [OUTW-1:0] fifo_in_data;
    logic fifo_in_valid;
    logic fifo_in_ready;
    
    // Instantiate input_mems module
    input_mems #(
        .INW(INW),
        .R(R),
        .C(C),
        .MAXK(MAXK)
    ) input_memory (
        .clk(clk),
        .reset(reset),
        .AXIS_TDATA(INPUT_TDATA),
        .AXIS_TVALID(INPUT_TVALID),
        .AXIS_TUSER(INPUT_TUSER),
        .AXIS_TREADY(INPUT_TREADY),
        .inputs_loaded(inputs_loaded),
        .compute_finished(compute_finished),
        .K(K),
        .B(B),
        .X_read_addr(X_read_addr),
        .X_data(X_data),
        .W_read_addr(W_read_addr),
        .W_data(W_data)
    );
    
    // Instantiate mac_pipe module
    mac_pipe #(
        .INW(INW),
        .OUTW(OUTW)
    ) mac (
        .clk(clk),
        .reset(reset),
        .input0(mac_input0),
        .input1(mac_input1),
        .init_value(mac_init_value),
        .out(mac_out),
        .init_acc(mac_init_acc),
        .input_valid(mac_input_valid)
    );
    
    // Instantiate fifo_out module
    fifo_out #(
        .OUTW(OUTW),
        .DEPTH(FIFO_DEPTH)
    ) output_fifo (
        .clk(clk),
        .reset(reset),
        .IN_AXIS_TDATA(fifo_in_data),
        .IN_AXIS_TVALID(fifo_in_valid),
        .IN_AXIS_TREADY(fifo_in_ready),
        .OUT_AXIS_TDATA(OUTPUT_TDATA),
        .OUT_AXIS_TVALID(OUTPUT_TVALID),
        .OUT_AXIS_TREADY(OUTPUT_TREADY)
    );
    
    // Control FSM and datapath
    // FSM states
    typedef enum logic [2:0] {
        IDLE,           // Wait for inputs_loaded
        INIT_ACC,       // Initialize accumulator with B
        SETUP_READ,     // Setup first memory read addresses (account for memory latency)
        COMPUTE,        // Compute K*K products for current output element
        WAIT_PIPE,      // Wait for pipeline to complete
        WRITE_FIFO,     // Write result to FIFO
        FINISH          // Set compute_finished signal
    } state_t;
    
    state_t state, state_next;
    
    // Counters for nested loops
    logic [R_BITS-1:0] r_cnt, r_cnt_next;        // Output row counter (0 to Rout-1)
    logic [C_BITS-1:0] c_cnt, c_cnt_next;        // Output column counter (0 to Cout-1)
    logic [K_CNT_BITS-1:0] i_cnt, i_cnt_next;    // Kernel row counter (0 to K-1)
    logic [K_CNT_BITS-1:0] j_cnt, j_cnt_next;    // Kernel column counter (0 to K-1)
    
    // Track how many MAC operations we've completed (for K*K loop)
    localparam MAC_CNT_BITS = $clog2(MAXK*MAXK+1);
    logic [MAC_CNT_BITS-1:0] mac_cnt, mac_cnt_next;
    
    // Pipeline wait counter (to account for 2-cycle MAC latency)
    logic [1:0] pipe_wait_cnt, pipe_wait_cnt_next;
    
    // Computed values
    logic [R_BITS-1:0] Rout;  // R - K + 1
    logic [C_BITS-1:0] Cout;  // C - K + 1
    
    assign Rout = R - K + 1;
    assign Cout = C - K + 1;
    
    // Address generation
    // X[r+i][c+j] maps to address: (r+i)*C + (c+j)
    logic [R_BITS:0] x_row;    // r + i (needs extra bit)
    logic [C_BITS:0] x_col;    // c + j (needs extra bit)
    assign x_row = r_cnt + i_cnt;
    assign x_col = c_cnt + j_cnt;
    assign X_read_addr = x_row * C + x_col;
    
    // W[i][j] maps to address: i*K + j
    assign W_read_addr = i_cnt * K + j_cnt;
    
    // Connect MAC inputs
    assign mac_input0 = X_data;
    assign mac_input1 = W_data;
    assign mac_init_value = B;
    
    // Connect FIFO input
    assign fifo_in_data = mac_out;
    
    // Combinational logic for FSM and counters
    always_comb begin
        // Default values
        state_next = state;
        r_cnt_next = r_cnt;
        c_cnt_next = c_cnt;
        i_cnt_next = i_cnt;
        j_cnt_next = j_cnt;
        mac_cnt_next = mac_cnt;
        pipe_wait_cnt_next = pipe_wait_cnt;
        
        // Default control signals
        mac_init_acc = 0;
        mac_input_valid = 0;
        fifo_in_valid = 0;
        compute_finished = 0;
        
        case (state)
            IDLE: begin
                // Wait for inputs to be loaded
                if (inputs_loaded) begin
                    state_next = INIT_ACC;
                    r_cnt_next = 0;
                    c_cnt_next = 0;
                end
            end
            
            INIT_ACC: begin
                // Initialize accumulator with bias B
                mac_init_acc = 1;
                // Setup first read addresses (0,0)
                i_cnt_next = 0;
                j_cnt_next = 0;
                mac_cnt_next = 0;
                state_next = SETUP_READ;
            end
            
            SETUP_READ: begin
                // Addresses (0,0) are being presented to memory
                // Memory will have data ready next cycle
                // Pre-increment to (0,1) for next fetch
                if (j_cnt == K - 1) begin
                    j_cnt_next = 0;
                    i_cnt_next = i_cnt + 1;
                end else begin
                    j_cnt_next = j_cnt + 1;
                end
                state_next = COMPUTE;
            end
            
            COMPUTE: begin
                // Use data from address presented TWO cycles ago
                // Current address (i_cnt, j_cnt) is being fetched for NEXT use
                mac_input_valid = 1;
                mac_cnt_next = mac_cnt + 1;
                
                // Pre-increment addresses for next fetch
                // (current addresses won't be used until next cycle)
                if (j_cnt == K - 1) begin
                    j_cnt_next = 0;
                    if (i_cnt == K - 1) begin
                        i_cnt_next = 0;
                    end else begin
                        i_cnt_next = i_cnt + 1;
                    end
                end else begin
                    j_cnt_next = j_cnt + 1;
                end
                
                // Check if we've USED K² elements
                if (mac_cnt == K * K - 1) begin
                    state_next = WAIT_PIPE;
                    pipe_wait_cnt_next = 0;
                    mac_cnt_next = 0;
                end
            end
            
            WAIT_PIPE: begin
                // Wait 2 cycles for MAC pipeline to complete
                // (1 cycle for multiply, 1 cycle for accumulate)
                if (pipe_wait_cnt == 1) begin
                    state_next = WRITE_FIFO;
                end else begin
                    pipe_wait_cnt_next = pipe_wait_cnt + 1;
                end
            end
            
            WRITE_FIFO: begin
                // Write MAC output to FIFO
                fifo_in_valid = 1;
                
                if (fifo_in_ready) begin
                    // Successfully wrote to FIFO
                    // Check if we're done with all output elements
                    if (c_cnt == Cout - 1) begin
                        c_cnt_next = 0;
                        if (r_cnt == Rout - 1) begin
                            // Done with all outputs
                            r_cnt_next = 0;
                            state_next = FINISH;
                        end else begin
                            // Move to next row
                            r_cnt_next = r_cnt + 1;
                            state_next = INIT_ACC;
                        end
                    end else begin
                        // Move to next column
                        c_cnt_next = c_cnt + 1;
                        state_next = INIT_ACC;
                    end
                end
                // else: FIFO full, stall in this state
            end
            
            FINISH: begin
                // Signal that computation is finished
                compute_finished = 1;
                state_next = IDLE;
            end
            
            default: begin
                state_next = IDLE;
            end
        endcase
    end
    
    // Sequential logic for state and counters
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            r_cnt <= 0;
            c_cnt <= 0;
            i_cnt <= 0;
            j_cnt <= 0;
            mac_cnt <= 0;
            pipe_wait_cnt <= 0;
        end else begin
            state <= state_next;
            r_cnt <= r_cnt_next;
            c_cnt <= c_cnt_next;
            i_cnt <= i_cnt_next;
            j_cnt <= j_cnt_next;
            mac_cnt <= mac_cnt_next;
            pipe_wait_cnt <= pipe_wait_cnt_next;
        end
    end

endmodule
