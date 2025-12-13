// Part 5 please somebody help me dear god
module mac_pipe #(
parameter INW = 16, 
parameter OUTW = 64,
parameter MULT_STAGES = 6
)(
input signed [INW-1:0] input0, input1, init_value,
output logic signed [OUTW-1:0] out,
input clk, reset, init_acc, input_valid
);

    // Register inputs to multiplier
    logic signed [INW-1:0] input0_reg, input1_reg;
    logic signed [2*INW-1:0] product;
    logic [MULT_STAGES:0] input_valid_pipe;

    // Input register
    always_ff @(posedge clk) begin
        if (reset) begin
            input0_reg <= 0;
            input1_reg <= 0;
        end else begin
            input0_reg <= input0;
            input1_reg <= input1;
        end
    end

    // generation block for DesignWare pipelined multipliers
    generate
        if (MULT_STAGES == 2) begin : g_mult2
            DW02_mult_2_stage #(.A_width(INW), .B_width(INW)) mult_i 
            (
                .A(input0_reg),
                .B(input1_reg),
                .TC(1'b1),
                .CLK(clk),
                .PRODUCT(product)
            );
        end else if (MULT_STAGES == 3) begin : g_mult3
            DW02_mult_3_stage #(.A_width(INW), .B_width(INW)) mult_i 
            (
                .A(input0_reg),
                .B(input1_reg),
                .TC(1'b1),
                .CLK(clk),
                .PRODUCT(product)
            );
        end else if (MULT_STAGES == 4) begin : g_mult4
            DW02_mult_4_stage #(.A_width(INW), .B_width(INW)) mult_i 
            (
                .A(input0_reg),
                .B(input1_reg),
                .TC(1'b1),
                .CLK(clk),
                .PRODUCT(product)
            );
        end else if (MULT_STAGES == 5) begin : g_mult5
            DW02_mult_5_stage #(.A_width(INW), .B_width(INW)) mult_i 
            (
                .A(input0_reg),
                .B(input1_reg),
                .TC(1'b1),
                .CLK(clk),
                .PRODUCT(product)
            );
        end else begin : g_mult6 // i set the pipeline stages # to 6 
            DW02_mult_6_stage #(.A_width(INW), .B_width(INW)) mult_i 
            (
                .A(input0_reg),
                .B(input1_reg),
                .TC(1'b1),
                .CLK(clk),
                .PRODUCT(product)
            );
        end
    endgenerate

    always_ff @(posedge clk) begin     // pipeline the valid bit to match mac speed
        if (reset) begin
            input_valid_pipe <= 0;
        end else begin
            input_valid_pipe <= {input_valid_pipe[MULT_STAGES-1:0], input_valid};
        end
    end

    // accumulate
    always_ff @(posedge clk) begin
        if (reset) begin
            out <= '0;
        end else if (init_acc) begin
            out <= init_value;
        end else if (input_valid_pipe[MULT_STAGES]) begin
            out <= out + product;
        end
    end
endmodule

module memory_dual_port #(
        parameter                WIDTH=24, 
        parameter                SIZE=19,
        localparam               LOGSIZE=$clog2(SIZE)
    )(
        input  [WIDTH-1:0]       data_in,
        output logic [WIDTH-1:0] data_out,
        input  [LOGSIZE-1:0]     write_addr, 
        input  [LOGSIZE-1:0]     read_addr,
        input                    clk, 
        input                    wr_en
    );

    logic [SIZE-1:0][WIDTH-1:0] mem;

    always_ff @(posedge clk) begin
        if (wr_en && (read_addr == write_addr))
            data_out <= data_in;
        else
            data_out <= mem[read_addr];

        if (wr_en)
            mem[write_addr] <= data_in;            
    end
endmodule

//part 2 is the same
module fifo_out #(
    parameter OUTW = 24,
    parameter DEPTH = 19,
    localparam LOGDEPTH = $clog2(DEPTH)
)(
    input clk,
    input reset,
    input [OUTW-1:0] IN_AXIS_TDATA,
    input IN_AXIS_TVALID,
    output logic IN_AXIS_TREADY,
    output logic [OUTW-1:0] OUT_AXIS_TDATA,
    output logic OUT_AXIS_TVALID,
    input OUT_AXIS_TREADY
);

    logic [LOGDEPTH-1:0] wr_ptr, rd_ptr;                            // read and write pointers
    logic wr_en, rd_en;                                             // read and write enable signals

    logic [LOGDEPTH-1:0] rd_ptr_next;
    logic [LOGDEPTH-1:0] mem_read_addr;                             // signals to look at the address one ahead

    logic [LOGDEPTH:0] capacity;
    logic fifo_full, fifo_empty;                                    // needed to know when we can read or write

    assign fifo_full = (capacity == DEPTH);
    assign fifo_empty = (capacity == 0);

    assign OUT_AXIS_TVALID = !fifo_empty;                           // AXI Stream interface signals
    assign IN_AXIS_TREADY = (!fifo_full || (fifo_full && rd_en)); 

    assign wr_en = IN_AXIS_TVALID && IN_AXIS_TREADY;               // read and write enable logic
    assign rd_en = OUT_AXIS_TVALID && OUT_AXIS_TREADY;

    always_comb begin
        if (rd_en) begin
            if (rd_ptr == DEPTH-1) begin
                rd_ptr_next = 0;                                    // tells us where the rd_ptr will point to in the next cycle (address + 1) so that memory module's data_out can get the correct value
            end else
                rd_ptr_next = rd_ptr + 1;
        end else begin
            rd_ptr_next = rd_ptr;
        end
    end

    assign mem_read_addr = rd_ptr_next; 

    memory_dual_port #(
        .SIZE (DEPTH),
        .WIDTH(OUTW)
    ) fifo_instance (
        .data_in (IN_AXIS_TDATA),
        .data_out (OUT_AXIS_TDATA),
        .write_addr (wr_ptr),
        .read_addr (mem_read_addr),
        .clk (clk),
        .wr_en (wr_en)
    );

    always_ff @(posedge clk) begin          // reset logic
        if (reset) begin
            wr_ptr   <= '0;
            rd_ptr   <= '0;
            capacity <= '0;
        end else begin
            if (wr_en) begin                // write logic
                if (wr_ptr == DEPTH-1)
                    wr_ptr <= '0;
                else
                    wr_ptr <= wr_ptr + 1;
            end

            rd_ptr <= rd_ptr_next;          // read logic (read pointer gets updated to rd_ptr_next)

            if (wr_en && !rd_en)            // capacity logic
                capacity <= capacity + 1;
            else if (!wr_en && rd_en)
                capacity <= capacity - 1;
            // if wr_en = 1 and rd_en = 1 its like nothing happened cap. wise
        end
    end
endmodule

module memory #(   
        parameter                   WIDTH=16, SIZE=64,
        localparam                  LOGSIZE=$clog2(SIZE)
    )(
        input [WIDTH-1:0]           data_in,
        output logic [WIDTH-1:0]    data_out,
        input [LOGSIZE-1:0]         addr,
        input                       clk, wr_en
    );

    logic [SIZE-1:0][WIDTH-1:0] mem;
    
    always_ff @(posedge clk) begin
        data_out <= mem[addr];
        if (wr_en)
            mem[addr] <= data_in;
    end
endmodule

//updated part 3 logic
module input_mems #(
    parameter INW = 24,
    parameter R = 9,
    parameter C = 8,
    parameter MAXK = 4,
    localparam K_BITS = $clog2(MAXK+1),
    localparam X_ADDR_BITS = $clog2(R*C),
    localparam W_ADDR_BITS = $clog2(MAXK*MAXK),
    parameter N_MACS = 4
)(
    input clk, reset,
    input  signed [INW-1:0] AXIS_TDATA,
    input  AXIS_TVALID,
    input  [K_BITS:0] AXIS_TUSER,   
    output logic AXIS_TREADY,
    output logic inputs_loaded,
    input compute_finished,
    output logic [K_BITS-1:0] K,
    output logic signed [INW-1:0] B,
    input  [X_ADDR_BITS-1:0] X_read_addr [N_MACS-1:0],
    output logic signed [INW-1:0] X_data [N_MACS-1:0],
    input  [W_ADDR_BITS-1:0] W_read_addr [N_MACS-1:0],
    output logic signed [INW-1:0] W_data [N_MACS-1:0]
);

    logic [K_BITS-1:0] TUSER_K;
    assign TUSER_K = AXIS_TUSER[K_BITS:1];
    logic new_W;
    assign new_W = AXIS_TUSER[0];
    
    // FInite State Machine initialization
    enum logic [2:0] {IDLE, LOAD_W, LOAD_B, LOAD_X, DONE} state, state_next;

    logic [K_BITS-1:0] K_reg, K_next;                   // signals that will hold K and B values in the current state and next state
    logic signed [INW-1:0] B_reg, B_next;

    logic [W_ADDR_BITS-1:0] w_index_reg, w_index_next;  // counters to iterate through memory addresses for W and X
    logic [X_ADDR_BITS-1:0] x_index_reg, x_index_next;

    localparam int X_LAST = R*C - 1;                    // the last address for X matrix (its R x C big)

    logic [W_ADDR_BITS-1:0] last_w_index;
    assign last_w_index = (K_reg * K_reg) - 1;          // the last address for W matrix (its K x K big)
            
    
    logic valid_and_ready;                              // AXI valid+ready handshake
    assign valid_and_ready = AXIS_TVALID && AXIS_TREADY;

    logic w_wr_en, x_wr_en;
    logic [W_ADDR_BITS-1:0] w_addr;
    logic [X_ADDR_BITS-1:0] x_addr;

    // Replicated memory outputs
    logic [INW-1:0] w_data_u [N_MACS-1:0];
    logic [INW-1:0] x_data_u [N_MACS-1:0];

    // generates copies of W and X memory for parallelizarion
    generate
        genvar m;
        for (m = 0; m < N_MACS; m++) begin
            memory #(.WIDTH(INW), .SIZE(MAXK*MAXK)) w_memory 
            (
                .clk(clk),
                .data_in(AXIS_TDATA),
                .data_out(w_data_u[m]),
                .addr(w_wr_en ? w_addr : W_read_addr[m]),
                .wr_en(w_wr_en)
            );

            memory #(.WIDTH(INW), .SIZE(R*C)) x_memory 
            (
                .clk(clk),
                .data_in(AXIS_TDATA),
                .data_out(x_data_u[m]),
                .addr(x_wr_en ? x_addr : X_read_addr[m]),
                .wr_en(x_wr_en)
            );

            always_comb begin
                W_data[m] = w_data_u[m];
                X_data[m] = x_data_u[m];
            end
        end
    endgenerate

    // FSM comb logic
    always_comb begin
        state_next = state;
        K_next = K_reg;     // regs that will hold value of K and B
        B_next = B_reg;
        w_index_next = w_index_reg; // index registers
        x_index_next = x_index_reg;

        AXIS_TREADY = 0;    // initialize singals
        inputs_loaded = 0;

        w_wr_en = 0;    // begin as not enabled
        x_wr_en = 0;
        w_addr = w_index_reg;
        x_addr = x_index_reg;

        case (state)
            IDLE: begin
                AXIS_TREADY = 1;            // set as ready
                if (valid_and_ready) begin
                    if (new_W) begin        // if new_W is 1, load the K and W vals 
                        K_next = TUSER_K;   // val of K
                        w_wr_en = 1;
                        w_addr = 0;         // write W[0]
                        w_index_next = 1;   // next W index
                        x_index_next = 0;
                        state_next = LOAD_W;
                    end else begin
                        x_wr_en = 1;    // if new_W = 0, just skip to load X
                        x_addr = 0;
                        x_index_next = 1;
                        state_next = LOAD_X;
                    end
                end
            end

            LOAD_W: begin
                AXIS_TREADY = 1;    // set as ready
                if (valid_and_ready) begin
                    w_wr_en = 1;
                    w_addr = w_index_reg;

                    if (w_index_reg == last_w_index) begin  // We done after K*K addresses
                        w_index_next = 0;                   // reset out w index
                        state_next = LOAD_B;
                    end else begin
                        w_index_next = w_index_reg + 1;  // iterate
                    end
                end
            end

            LOAD_B: begin
                AXIS_TREADY = 1;    // set as ready
                if (valid_and_ready) begin
                    B_next = AXIS_TDATA;    // load B
                    x_index_next = 0;       // start X at address 0
                    state_next = LOAD_X;
                end
            end

            LOAD_X: begin
                AXIS_TREADY = 1;    // set as ready
                if (valid_and_ready) begin
                    x_wr_en = 1;
                    x_addr = x_index_reg;

                    if (x_index_reg == X_LAST[X_ADDR_BITS-1:0]) begin
                        x_index_next = 0;
                        state_next = DONE;
                    end else begin
                        x_index_next = x_index_reg + 1;
                    end
                end
            end

            DONE: begin
                AXIS_TREADY = 0;    // not ready to take more data
                inputs_loaded = 1;  // we done loading everything
                if (compute_finished) begin
                    state_next = IDLE;  // go back to idle
                    w_index_next = 0;   // reset indexes
                    x_index_next = 0;
                end
            end

            default: state_next = IDLE;
        endcase
    end

    always_ff @(posedge clk) begin  // update the states of everything
        if (reset) begin
            state <= IDLE;
            K_reg <= 0;
            B_reg <= 0;
            w_index_reg <= 0;
            x_index_reg <= 0;
        end else begin
            state <= state_next;
            K_reg <= K_next;
            B_reg <= B_next;
            w_index_reg <= w_index_next;
            x_index_reg <= x_index_next;
        end
    end

    // assign vals to outputs
    assign K = K_reg;
    assign B = B_reg;

endmodule


// Accelerated conv >:)
module Conv #(
    parameter INW = 18,
    parameter R = 8,
    parameter C = 8,
    parameter MAXK = 5,
    localparam OUTW = $clog2(MAXK*MAXK*(128'd1 << 2*INW-2) + (1<<(INW-1)))+1,
    localparam K_BITS = $clog2(MAXK+1),
    parameter MULT_STAGES = 6,
    parameter N_MACS = 4
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
    // new parameters
    localparam MAC_LATENCY = MULT_STAGES + 1;  // add one cycle to account for accumulator
    localparam MAC_WAIT_BITS = $clog2(MAC_LATENCY + 2);
    localparam MAC_INDEX_BITS = $clog2(MAXK*MAXK + 1);

    // signals from parts 1-3 
    logic inputs_loaded;
    logic compute_finished;
    logic [K_BITS-1:0] K;
    logic signed [INW-1:0] B;

    // Per-MAC memory ports
    logic [X_ADDR_BITS-1:0] X_read_addr [N_MACS-1:0];
    logic signed [INW-1:0] X_data [N_MACS-1:0];
    logic [W_ADDR_BITS-1:0] W_read_addr [N_MACS-1:0];
    logic signed [INW-1:0] W_data [N_MACS-1:0];

    logic [OUTW-1:0] fifo_in_data;
    logic fifo_in_valid;
    logic fifo_in_ready;

    logic mac_init_acc;
    logic mac_input_valid [N_MACS-1:0];
    logic signed [OUTW-1:0] mac_out [N_MACS-1:0];

    // Instantiate legal replicated memories
    input_mems #(.INW(INW), .R(R), .C(C), .MAXK(MAXK), .N_MACS(N_MACS)) input_memory 
    (
        .clk             (clk),
        .reset           (reset),
        .AXIS_TDATA      (INPUT_TDATA),
        .AXIS_TVALID     (INPUT_TVALID),
        .AXIS_TUSER      (INPUT_TUSER),
        .AXIS_TREADY     (INPUT_TREADY),
        .inputs_loaded   (inputs_loaded),
        .compute_finished(compute_finished),
        .K               (K),
        .B               (B),
        .X_read_addr     (X_read_addr),
        .X_data          (X_data),
        .W_read_addr     (W_read_addr),
        .W_data          (W_data)
    );

    fifo_out #(.OUTW(OUTW), .DEPTH(FIFO_DEPTH)) output_fifo 
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

    // B value goes to MAC[0] and the others just get zeros
    logic signed [INW-1:0] mac_init_values [N_MACS-1:0];
    always_comb begin
        mac_init_values[0] = B;
        for (int m = 1; m < N_MACS; m++)
            mac_init_values[m] = 0;
    end

    // Generate [N_MACS] pipelines
    genvar g;
    generate
        for (g = 0; g < N_MACS; g++) begin : MACS
            mac_pipe #(.INW(INW), .OUTW(OUTW), .MULT_STAGES(MULT_STAGES)) mac_i 
            (
                .clk(clk),
                .reset(reset),
                .input0(X_data[g]),
                .input1(W_data[g]),
                .init_value(mac_init_values[g]),
                .init_acc(mac_init_acc),
                .input_valid(mac_input_valid[g]),
                .out(mac_out[g])
            );
        end
    endgenerate

    // sum the MAC outputs 
    logic signed [OUTW-1:0] mac_sum;
    always_comb begin
        mac_sum = mac_out[0];
        for (int m = 1; m < N_MACS; m++)
            mac_sum += mac_out[m];
    end
    assign fifo_in_data = mac_sum;

    enum logic [2:0] {IDLE, INIT, READ, COMPUTE, WAIT, WRITE, DONE} state, state_next;

    // Output indices
    logic [R_BITS-1:0] r_index, r_index_next;
    logic [C_BITS-1:0] c_index, c_index_next;

    // kernel indices for each MAC
    logic [K_INDEX_BITS-1:0] i_index [N_MACS-1:0];
    logic [K_INDEX_BITS-1:0] j_index [N_MACS-1:0];
    logic [K_INDEX_BITS-1:0] i_index_next [N_MACS-1:0];
    logic [K_INDEX_BITS-1:0] j_index_next [N_MACS-1:0];

    logic [MAC_INDEX_BITS-1:0] mac_iter, mac_iter_next; // iterates to K^2 / NMACS
    logic [MAC_WAIT_BITS-1:0] wait_count, wait_count_next; // counts cycles after the last MAC input_valid to wait for pipeline to finish

    logic [R_BITS-1:0] Rout;
    logic [C_BITS-1:0] Cout;
    assign Rout = R - K + 1;
    assign Cout = C - K + 1;

    logic [MAC_INDEX_BITS-1:0] total_ops; //total # of MAC operations to compute 1 output
    assign total_ops = K*K;

    // ceil(total_ops / N_MACS)
    logic [MAC_INDEX_BITS+3:0] total_ops_padded;
    logic [MAC_INDEX_BITS-1:0] num_iterations;
    assign total_ops_padded = {4'b0, total_ops} + (N_MACS - 1);
    assign num_iterations = total_ops_padded / N_MACS;

    //I am using N_MACs = 4 so we need to compute multiples of K (x2 and x3) in order to advance to the to next row
    logic [K_BITS+1:0] Kx2, Kx3;
    assign Kx2 = K << 1;
    assign Kx3 = K + (K << 1);

    logic [K_INDEX_BITS+1:0] new_j_temp [N_MACS-1:0];   // The column index after adding N_MACS
    logic [K_INDEX_BITS:0] next_i_temp [N_MACS-1:0];   // next row index
    logic [K_INDEX_BITS-1:0] next_j_temp [N_MACS-1:0]; //wrapped column index

    // address generation
    always_comb begin
        for (int m = 0; m < N_MACS; m++) begin
            X_read_addr[m] = (r_index + i_index[m]) * C + (c_index + j_index[m]);
            W_read_addr[m] = i_index[m] * K + j_index[m];
        end
    end

    //FSM comb logic block
    always_comb begin
        state_next = state; 
        r_index_next = r_index;
        c_index_next = c_index;
        mac_iter_next = mac_iter;
        wait_count_next = wait_count;

        mac_init_acc = 0;
        fifo_in_valid = 0;
        compute_finished = 0;

        for (int m = 0; m < N_MACS; m++) begin //initialize everythig
            mac_input_valid[m] = 0;
            i_index_next[m] = i_index[m];
            j_index_next[m] = j_index[m];
            new_j_temp[m] = 0;
            next_i_temp[m] = 0;
            next_j_temp[m] = 0;
        end

        case (state)    
            IDLE: begin
                if (inputs_loaded) begin
                    r_index_next = 0;
                    c_index_next = 0;
                    mac_iter_next = 0;
                    state_next = INIT;
                end
            end

            INIT: begin
                mac_init_acc = 1;
                mac_iter_next = 0;
                for (int m = 0; m < N_MACS; m++) begin  // initialize each MAC to index m
                    if (m < total_ops) begin
                        i_index_next[m] = m / K;
                        j_index_next[m] = m % K;
                    end else begin
                        i_index_next[m] = 0;
                        j_index_next[m] = 0;
                    end
                end

                state_next = READ;
            end

            READ: begin
                //wait a cycle to read
                state_next = COMPUTE;
            end

            COMPUTE: begin
                for (int m = 0; m < N_MACS; m++) begin
                    if ((mac_iter * N_MACS + m) < total_ops)
                        mac_input_valid[m] = 1;
                end
                mac_iter_next = mac_iter + 1;

                // iterate each MAC's indexes by N_MACS positions 
                for (int m = 0; m < N_MACS; m++) begin
                    new_j_temp[m] = j_index[m] + N_MACS;

                    if (new_j_temp[m] >= Kx3) begin // advance 3 rows (new_j >= 3*K)
                        next_j_temp[m] = new_j_temp[m] - Kx3;
                        next_i_temp[m] = i_index[m] + 3;
                    end else if (new_j_temp[m] >= Kx2) begin // advance 2 rows (new_j >= 2*K)
                        next_j_temp[m] = new_j_temp[m] - Kx2;
                        next_i_temp[m] = i_index[m] + 2;
                    end else if (new_j_temp[m] >= K) begin // advance 1 row (new_j >= K)
                        next_j_temp[m] = new_j_temp[m] - K;
                        next_i_temp[m] = i_index[m] + 1;
                    end else begin                          //advance in the same row
                        next_j_temp[m] = new_j_temp[m];
                        next_i_temp[m] = i_index[m];
                    end

                    i_index_next[m] = next_i_temp[m][K_INDEX_BITS-1:0]; 
                    j_index_next[m] = next_j_temp[m];
                end

                if (mac_iter >= num_iterations - 1) begin // all kernel products finished going into the MACs
                    state_next = WAIT;
                    wait_count_next = 0;
                end
            end

            WAIT: begin 
                if (wait_count >= MAC_LATENCY-1) begin // need to wait based on pipeline stages + accumulator cycle
                    state_next = WRITE;
                end else begin
                    wait_count_next = wait_count + 1;
                end
            end

            WRITE: begin
                fifo_in_valid = 1;  //push the convolution result into the output FIFO
                if (fifo_in_ready) begin
                    if (c_index == (Cout-1)) begin // checks if we are at the last column for the current row
                        c_index_next = 0;
                        if (r_index == (Rout-1)) begin // check if we are at the last row
                            state_next = DONE;
                        end else begin
                            r_index_next  = r_index + 1; // move to the next row
                            mac_iter_next = 0;      // Reset mac counter for the next window
                            state_next = INIT;
                        end
                    end else begin
                        c_index_next = c_index + 1; // Move to the next column in the same row
                        mac_iter_next = '0;
                        state_next = INIT;
                    end
                end
            end

            DONE: begin             // we finished computing yipee
                compute_finished = 1;
                state_next = IDLE;
            end

            default: state_next = IDLE;
        endcase
    end

    // FSM seq logic
    always_ff @(posedge clk) begin
        if (reset) begin //reset logic
            state <= IDLE;
            r_index <= 0;
            c_index <= 0; // reset to position (0,0)
            mac_iter <= 0;
            wait_count <= 0;
            for (int m = 0; m < N_MACS; m++) begin //resets the indices for each mac
                i_index[m] <= 0;
                j_index[m] <= 0;
            end
        end else begin
            state      <= state_next;
            r_index    <= r_index_next; // update window position
            c_index    <= c_index_next;
            mac_iter   <= mac_iter_next; //update mac iterate counter
            wait_count <= wait_count_next; //update the wait cointer
            for (int m = 0; m < N_MACS; m++) begin //update indices of each mac
                i_index[m] <= i_index_next[m];
                j_index[m] <= j_index_next[m];
            end
        end
    end
endmodule 
