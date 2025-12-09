//*** OPTIMIZED CONV MODULE FOR PART 5 - VERSION 1 ***
//*** Optimization: Deeper pipelining with DesignWare pipelined multiplier ***

//Part 1: MAC with Pipelined DesignWare Multiplier
module mac_pipe #(
    parameter INW = 16, 
    parameter OUTW = 64,
    parameter MULT_STAGES = 5   // Number of pipeline stages in the multiplier (2-6)
)(
    input signed [INW-1:0] input0, input1, init_value,
    output logic signed [OUTW-1:0] out,
    input clk, reset, init_acc, input_valid
);

    // Input registers for better timing
    logic signed [INW-1:0] input0_reg, input1_reg;
    
    // Product output from pipelined multiplier
    logic signed [INW*2-1:0] product;
    
    // Delayed valid signal to match pipeline depth
    logic [MULT_STAGES:0] input_valid_pipe;
    
    // Register inputs for better timing
    always_ff @(posedge clk) begin
        if (reset) begin
            input0_reg <= 0;
            input1_reg <= 0;
        end else begin
            input0_reg <= input0;
            input1_reg <= input1;
        end
    end
    
    // Instantiate DesignWare pipelined multiplier
    // DW02_mult_3_stage for 3 pipeline stages
    // Adjust the number in the module name (2-6) to change pipeline depth
    generate
        if (MULT_STAGES == 2) begin : mult_2stage
            DW02_mult_2_stage #(INW, INW) mult_inst (
                .A(input0_reg),
                .B(input1_reg),
                .TC(1'b1),  // Two's complement
                .CLK(clk),
                .PRODUCT(product)
            );
        end else if (MULT_STAGES == 3) begin : mult_3stage
            DW02_mult_3_stage #(INW, INW) mult_inst (
                .A(input0_reg),
                .B(input1_reg),
                .TC(1'b1),
                .CLK(clk),
                .PRODUCT(product)
            );
        end else if (MULT_STAGES == 4) begin : mult_4stage
            DW02_mult_4_stage #(INW, INW) mult_inst (
                .A(input0_reg),
                .B(input1_reg),
                .TC(1'b1),
                .CLK(clk),
                .PRODUCT(product)
            );
        end else if (MULT_STAGES == 5) begin : mult_5stage
            DW02_mult_5_stage #(INW, INW) mult_inst (
                .A(input0_reg),
                .B(input1_reg),
                .TC(1'b1),
                .CLK(clk),
                .PRODUCT(product)
            );
        end else if (MULT_STAGES == 6) begin : mult_6stage
            DW02_mult_6_stage #(INW, INW) mult_inst (
                .A(input0_reg),
                .B(input1_reg),
                .TC(1'b1),
                .CLK(clk),
                .PRODUCT(product)
            );
        end
    endgenerate
    
    // Pipeline the input_valid signal to match the multiplier latency
    // Need to account for: input register (1) + multiplier stages (MULT_STAGES)
    always_ff @(posedge clk) begin
        if (reset) begin
            input_valid_pipe <= 0;
        end else begin
            input_valid_pipe <= {input_valid_pipe[MULT_STAGES-1:0], input_valid};
        end
    end
    
    // Accumulator - now uses the output from the pipelined multiplier
    always_ff @(posedge clk) begin
        if (reset) begin
            out <= 0;
        end else if (init_acc) begin
            out <= init_value;
        end else if (input_valid_pipe[MULT_STAGES]) begin
            out <= product + out;
        end else begin
            out <= out;
        end      
    end
endmodule

//--------------------------------------------------------------------------------------------------------------
//Part 2: (unchanged from Part 4)
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

    logic [LOGDEPTH-1:0] wr_ptr, rd_ptr;
    logic wr_en, rd_en;

    logic [LOGDEPTH-1:0] rd_ptr_next;
    logic [LOGDEPTH-1:0] mem_read_addr;

    logic [LOGDEPTH:0] capacity;
    logic fifo_full, fifo_empty;

    assign fifo_full = (capacity == DEPTH);
    assign fifo_empty = (capacity == 0);

    assign OUT_AXIS_TVALID = !fifo_empty;
    assign IN_AXIS_TREADY = (!fifo_full || (fifo_full && rd_en)); 

    assign wr_en = IN_AXIS_TVALID && IN_AXIS_TREADY;
    assign rd_en = OUT_AXIS_TVALID && OUT_AXIS_TREADY;

    always_comb begin
        if (rd_en) begin
            if (rd_ptr == DEPTH-1) begin
                rd_ptr_next = 0;
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

    always_ff @(posedge clk) begin
        if (reset) begin
            wr_ptr   <= '0;
            rd_ptr   <= '0;
            capacity <= '0;
        end else begin
            if (wr_en) begin
                if (wr_ptr == DEPTH-1)
                    wr_ptr <= '0;
                else
                    wr_ptr <= wr_ptr + 1;
            end

            rd_ptr <= rd_ptr_next;

            if (wr_en && !rd_en)
                capacity <= capacity + 1;
            else if (!wr_en && rd_en)
                capacity <= capacity - 1;
        end
    end
endmodule

//--------------------------------------------------------------------------------------------------------------
//Part 3: (unchanged from Part 4)
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

module input_mems #(
    parameter INW = 24,
    parameter R = 9,
    parameter C = 8,
    parameter MAXK = 4,
    localparam K_BITS = $clog2(MAXK+1),
    localparam X_ADDR_BITS = $clog2(R*C),
    localparam W_ADDR_BITS = $clog2(MAXK*MAXK)
)(
    input clk, reset,
    input [INW-1:0] AXIS_TDATA,
    input AXIS_TVALID,
    input [K_BITS:0] AXIS_TUSER,
    output logic AXIS_TREADY,
    output logic inputs_loaded,
    input compute_finished,
    output logic [K_BITS-1:0] K,
    output logic signed [INW-1:0] B,
    input [X_ADDR_BITS-1:0] X_read_addr,
    output logic signed [INW-1:0] X_data,
    input [W_ADDR_BITS-1:0] W_read_addr,
    output logic signed [INW-1:0] W_data
);

    logic [K_BITS-1:0] TUSER_K;
    assign TUSER_K = AXIS_TUSER[K_BITS:1];
    logic new_W;
    assign new_W = AXIS_TUSER[0];

    logic [INW-1:0] w_data_out, x_data_out;
    logic [W_ADDR_BITS-1:0] w_addr;
    logic [X_ADDR_BITS-1:0] x_addr;
    logic w_wr_en, x_wr_en;

    memory #(.WIDTH(INW), .SIZE(MAXK*MAXK)) w_memory
    ( 
        .clk    (clk),
        .data_in(AXIS_TDATA),
        .data_out(w_data_out),
        .addr   (w_addr),
        .wr_en  (w_wr_en)
    );

    memory #(.WIDTH(INW), .SIZE(R*C)) x_memory 
    (
        .clk (clk),
        .data_in (AXIS_TDATA),
        .data_out (x_data_out),
        .addr (x_addr),
        .wr_en (x_wr_en)
    );

    assign W_data = w_data_out;
    assign X_data = x_data_out;

    enum logic [2:0] {IDLE, LOAD_W, LOAD_B, LOAD_X, DONE} state, state_next;

    logic [K_BITS-1:0] K_reg, K_next;
    logic signed [INW-1:0] B_reg, B_next;

    logic [W_ADDR_BITS-1:0] w_index_reg, w_index_next;
    logic [X_ADDR_BITS-1:0] x_index_reg, x_index_next;

    localparam int X_LAST = R*C - 1;

    logic [W_ADDR_BITS-1:0] last_w_index;
    assign last_w_index = (K_reg * K_reg) - 1;
            
    logic valid_and_ready;
    assign valid_and_ready = AXIS_TVALID && AXIS_TREADY;

    always_comb begin
        state_next = state;
        K_next = K_reg;
        B_next = B_reg;
        w_index_next = w_index_reg;
        x_index_next = x_index_reg;

        AXIS_TREADY = 0;
        inputs_loaded = 0;

        w_wr_en = 0;
        x_wr_en = 0;

        w_addr = (state == DONE) ? W_read_addr : w_index_reg;
        x_addr = (state == DONE) ? X_read_addr : x_index_reg;

        case (state)
            IDLE: begin
                AXIS_TREADY = 1;
                if (valid_and_ready) begin
                    if (new_W) begin
                        K_next = TUSER_K;
                        w_wr_en = 1;
                        w_addr = 0;
                        w_index_next = 1;
                        x_index_next = 0;
                        state_next = LOAD_W;
                    end
                    else begin
                        x_wr_en = 1;
                        x_addr = 0; 
                        x_index_next = 1;
                        state_next = LOAD_X;
                    end
                end
            end

            LOAD_W: begin
                AXIS_TREADY = 1;
                if (valid_and_ready) begin
                    w_wr_en = 1;
                    w_addr = w_index_reg;

                    if (w_index_reg == last_w_index) begin
                        w_index_next = 0;
                        state_next = LOAD_B;
                    end
                    else begin
                        w_index_next = w_index_reg + 1;
                    end
                end
            end

            LOAD_B: begin
                AXIS_TREADY = 1;
                if (valid_and_ready) begin
                    B_next = AXIS_TDATA;
                    x_index_next = 0;
                    state_next = LOAD_X;
                end
            end 

            LOAD_X: begin
                AXIS_TREADY = 1;
                if (valid_and_ready) begin
                    x_wr_en = 1;
                    x_addr = x_index_reg;

                    if (x_index_reg == X_LAST[X_ADDR_BITS-1:0]) begin
                        x_index_next = 0;
                        state_next = DONE;
                    end
                    else begin
                        x_index_next = x_index_reg + 1;
                    end
                end
            end

            DONE: begin
                AXIS_TREADY = 0;
                inputs_loaded = 1;

                if (compute_finished) begin
                    state_next = IDLE;
                    w_index_next = 0;
                    x_index_next = 0; 
                end
            end

            default: begin
                state_next = IDLE;
            end
        endcase
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            K_reg <= '0;
            B_reg <= '0;
            w_index_reg <= '0;
            x_index_reg <= '0;
        end else begin
            state <= state_next;
            K_reg <= K_next;
            B_reg <= B_next;
            w_index_reg <= w_index_next;
            x_index_reg <= x_index_next;
        end
    end

    assign K = K_reg;
    assign B = B_reg;

endmodule

//--------------------------------------------------------------------------------------------------------------
//Part 4: Conv Module with Updated Control for Deeper Pipeline
module Conv #(
    parameter INW = 18,
    parameter R = 8,
    parameter C = 8,
    parameter MAXK = 5,
    parameter MULT_STAGES = 3,  // Pipeline stages for multiplier
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

    localparam FIFO_DEPTH = C - 1;
    localparam X_ADDR_BITS = $clog2(R * C);
    localparam W_ADDR_BITS = $clog2(MAXK * MAXK);
    localparam R_BITS = $clog2(R);
    localparam C_BITS = $clog2(C);
    localparam K_INDEX_BITS = $clog2(MAXK);
    
    // Calculate MAC latency: input_reg (1) + mult_stages (MULT_STAGES) + accum (1)
    localparam MAC_LATENCY = MULT_STAGES + 1;
    localparam MAC_WAIT_BITS = $clog2(MAC_LATENCY + 1);
    
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

    input_mems #(.INW(INW), .R(R), .C(C), .MAXK(MAXK)) input_memory
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
    
    mac_pipe #(.INW(INW), .OUTW(OUTW), .MULT_STAGES(MULT_STAGES)) mac
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
    
    enum logic [2:0] {IDLE, SET_MAC, READ, COMPUTE, WAIT, WRITE, DONE} state, state_next;
    
    logic [R_BITS-1:0] r_index, r_index_next;
    logic [C_BITS-1:0] c_index, c_index_next;      
    logic [K_INDEX_BITS-1:0] i_index, i_index_next;
    logic [K_INDEX_BITS-1:0] j_index, j_index_next;   
    
    localparam MAC_INDEX_BITS = $clog2(MAXK*MAXK+1);
    logic [MAC_INDEX_BITS-1:0] mac_index, mac_index_next;
    
    // Updated to handle longer pipeline
    logic [MAC_WAIT_BITS-1:0] mac_wait_index, mac_wait_index_next;
    
    logic [R_BITS-1:0] Rout; 
    logic [C_BITS-1:0] Cout;
    assign Rout = R - K + 1;
    assign Cout = C - K + 1;
    
    logic [R_BITS:0] x_row;   
    logic [C_BITS:0] x_col;   
    assign x_row = r_index + i_index;
    assign x_col = c_index + j_index;

    assign X_read_addr = x_row * C + x_col;
    assign W_read_addr = i_index * K + j_index;
    
    assign mac_input0 = X_data;
    assign mac_input1 = W_data;
    assign mac_init_value = B;
    
    assign fifo_in_data = mac_out;
    
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
            IDLE: begin
                if (inputs_loaded) begin
                    r_index_next = 0;
                    c_index_next = 0;
                    state_next = SET_MAC;
                end
            end
            
            SET_MAC: begin
                mac_init_acc = 1;
                i_index_next = 0;
                j_index_next = 0;
                mac_index_next = 0;
                state_next = READ;
            end
            
            READ: begin
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

                if (j_index == K - 1) begin
                    j_index_next = 0;
                    if (i_index == K - 1) begin
                        i_index_next = 0;
                    end else begin
                        i_index_next = i_index + 1;
                    end
                end else begin
                    j_index_next = j_index + 1;
                end
                        
                if (mac_index == K * K - 1) begin
                    state_next = WAIT;
                    mac_wait_index_next = 0;
                    mac_index_next = 0;
                end
            end
            
            WAIT: begin
                // Wait for MAC_LATENCY - 1 cycles
                if (mac_wait_index == MAC_LATENCY - 1) begin
                    state_next = WRITE;
                end else begin
                    mac_wait_index_next = mac_wait_index + 1;
                end
            end
            
            WRITE: begin
                fifo_in_valid = 1;
                
                if (fifo_in_ready) begin
                    if (c_index == Cout - 1) begin
                        c_index_next = 0;
                        if (r_index == Rout - 1) begin
                            r_index_next = 0;
                            state_next = DONE;
                        end else begin
                            r_index_next = r_index + 1;
                            state_next = SET_MAC;
                        end
                    end else begin
                        c_index_next = c_index + 1;
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
    
    always_ff @(posedge clk) begin
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
