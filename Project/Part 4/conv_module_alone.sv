//Part 4:
module Conv #(
    parameter INW = 18,
    parameter R = 8,
    parameter C = 8,
    parameter MAXK = 5,
    localparam OUTW = $clog2(MAXK*MAXK*(128'd1 << 2*INW-2) + (1<<(INW-1)))+1,
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

    // parameters from our parts 1-3
    localparam FIFO_DEPTH = C - 1;
    localparam X_ADDR_BITS = $clog2(R * C);
    localparam W_ADDR_BITS = $clog2(MAXK * MAXK);
    localparam R_BITS = $clog2(R);
    localparam C_BITS = $clog2(C);
    localparam K_INDEX_BITS = $clog2(MAXK);
    
    // signals from parts 1-3 
    logic inputs_loaded;
    logic compute_finished;
    
    logic [K_BITS-1:0] K;
    logic signed [INW-1:0] B;
    
    logic [X_ADDR_BITS-1:0] X_read_addr;
    logic signed [INW-1:0] X_data;
    logic [W_ADDR_BITS-1:0] W_read_addr;
    logic signed [INW-1:0] W_data;

    logic [OUTW-1:0] fifo_in_data;
    logic fifo_in_valid;
    logic fifo_in_ready;
    
    logic signed [INW-1:0] mac_input0, mac_input1, mac_init_value;
    logic mac_init_acc, mac_input_valid;
     logic signed [OUTW-1:0] mac_out;

    input_mems #(.INW(INW), .R(R), .C(C), .MAXK(MAXK)) input_memory // instantiate input_mems (part 3)
    (
        .clk(clk),
        .reset(reset),
        .AXIS_TDATA(INPUT_TDATA),
        .AXIS_TVALID(INPUT_TVALID),
        .AXIS_TUSER(INPUT_TUSER),
        .AXIS_TREADY(INPUT_TREADY),
        .K(K),
        .B(B),
        .X_read_addr(X_read_addr),
        .X_data(X_data),
        .W_read_addr(W_read_addr),
        .W_data(W_data),
        .inputs_loaded(inputs_loaded),
        .compute_finished(compute_finished)
    );

    
    fifo_out #(.OUTW(OUTW), .DEPTH(FIFO_DEPTH)) output_fifo // instantiate fifo_out (part 2))
    (
        .clk(clk),
        .reset(reset),
        .IN_AXIS_TDATA(fifo_in_data),
        .IN_AXIS_TVALID(fifo_in_valid),
        .IN_AXIS_TREADY(fifo_in_ready),
        .OUT_AXIS_TDATA(OUTPUT_TDATA),
        .OUT_AXIS_TVALID(OUTPUT_TVALID),
        .OUT_AXIS_TREADY(OUTPUT_TREADY)
    );
    
    mac_pipe #(.INW(INW),.OUTW(OUTW)) mac   // instantiate mac_pipe (part1)
    (
        .clk(clk),
        .reset(reset),
        .input0(mac_input0),
        .input1(mac_input1),
        .init_value(mac_init_value),
        .out(mac_out),
        .init_acc(mac_init_acc),
        .input_valid(mac_input_valid)
    );
    
    // FInite State Machine initialization
    enum logic [2:0] {IDLE, SET_MAC, READ, COMPUTE, WAIT, WRITE, DONE} state, state_next;
    
    // Counters for nested loops
    logic [R_BITS-1:0] r_index, r_index_next;        // iterators for R rows and C columns
    logic [C_BITS-1:0] c_index, c_index_next;      
    logic [K_INDEX_BITS-1:0] i_index, i_index_next;    // iterators for ith rows  and jth columns
    logic [K_INDEX_BITS-1:0] j_index, j_index_next;   
    
    localparam MAC_INDEX_BITS = $clog2(MAXK*MAXK+1); // # of mac ops we need to compute
    logic [MAC_INDEX_BITS-1:0] mac_index, mac_index_next;
    
    logic [1:0] mac_wait_index, mac_wait_index_next; // ignals needed to wait for the mac to compute
    
    // X matrix outputs
    logic [R_BITS-1:0] Rout; 
    logic [C_BITS-1:0] Cout;
    assign Rout = R - K + 1;
    assign Cout = C - K + 1;
    
    logic [R_BITS:0] x_row;   
    logic [C_BITS:0] x_col;   
    assign x_row = r_index + i_index;
    assign x_col = c_index + j_index;

    assign X_read_addr = x_row * C + x_col;    // X[r+i][c+j] -> C(r+i) + (c+j) (big C is the coulumn constant and little c is the iterator)
    assign W_read_addr = i_index * K + j_index;   // W[i][j] -> Ki + j
    
    assign mac_input0 = X_data;     // connect data to our mac module
    assign mac_input1 = W_data;
    assign mac_init_value = B;
    
    assign fifo_in_data = mac_out; // mac outputs go to inputs of fifo
    
    // FSM comblogic block
    always_comb begin
        state_next = state;
        r_index_next = r_index;
        c_index_next = c_index;
        
        i_index_next = i_index;
        j_index_next = j_index;
        
        mac_index_next = mac_index;
        mac_wait_index_next = mac_wait_index;
        mac_input_valid = 0;
        mac_init_acc = 0;
       
        fifo_in_valid = 0;
        compute_finished = 0;
        
        case (state)
            IDLE: begin                 // wait for inputs to be loaded
                if (inputs_loaded) begin
                    r_index_next = 0;
                    c_index_next = 0;
                    state_next = SET_MAC;
                end
            end
            
            SET_MAC: begin
                mac_init_acc = 1;   // set the mac with defautl vals
                i_index_next = 0;
                j_index_next = 0;
                mac_index_next = 0;
                state_next = READ;
            end
            
            READ: begin                     // read mem addresses
                if (j_index == K - 1) begin
                    j_index_next = 0;
                    i_index_next = i_index + 1;
                end else begin
                    j_index_next = j_index + 1;
                end
                state_next = COMPUTE;
            end
            
            COMPUTE: begin
                mac_input_valid = 1;
                mac_index_next = mac_index + 1;

                if (j_index == K - 1) begin // addr i,j will be used next cycle
                    j_index_next = 0;
                    if (i_index == K - 1) begin
                        i_index_next = 0;
                    end else begin
                        i_index_next = i_index + 1;
                    end
                end else begin
                    j_index_next = j_index + 1;
                end
                        
                if (mac_index == K * K - 1) begin   // go to next state after going thru the W matrix
                    state_next = WAIT;
                    mac_wait_index_next = 0;
                    mac_index_next = 0;
                end
            end
            
            WAIT: begin                         // we need to wait 1 cycle to multiply and another to add so 2 total)
                if (mac_wait_index == 1) begin
                    state_next = WRITE;
                end else begin
                    mac_wait_index_next = mac_wait_index + 1;
                end
            end
            
            WRITE: begin               // write MAC outputs to our fifo
                fifo_in_valid = 1;
                
                if (fifo_in_ready) begin
                    if (c_index == Cout - 1) begin
                        c_index_next = 0;
                        if (r_index == Rout - 1) begin
                            r_index_next = 0;
                            state_next = DONE;
                        end else begin
                            r_index_next = r_index + 1; // iterates thru rows
                            state_next = SET_MAC;
                        end
                    end else begin
                        c_index_next = c_index + 1; // iterates thru col
                        state_next = SET_MAC;
                    end
                end
            end
            
            DONE: begin
                compute_finished = 1;
                state_next = IDLE;
            end
            
            default: begin
                state_next = IDLE;
            end
        endcase
    end
    
    always_ff @(posedge clk) begin      // update the states of everything
        if (reset) begin
            r_index <= 0;
            c_index <= 0;
            i_index <= 0;
            j_index <= 0;
            
            mac_index <= 0;
            mac_wait_index <= 0;
            
            state <= IDLE;
        end else begin
            r_index <= r_index_next;
            c_index <= c_index_next;
            i_index <= i_index_next;
            j_index <= j_index_next;
            
            mac_index <= mac_index_next;
            mac_wait_index <= mac_wait_index_next;
            
            state <= state_next;
        end
    end
endmodule
