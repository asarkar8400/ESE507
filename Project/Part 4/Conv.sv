//*** CONV MODULE START AT LINE 368 ***
//Part 1:
module mac_pipe #(
parameter INW = 16, 
parameter OUTW = 64
)(
input signed [INW-1:0] input0, input1, init_value,
output logic signed [OUTW-1:0] out,
input clk, reset, init_acc, input_valid
);

// logic signed [OUTW-1:0] product, q;
// edited based on prof feedback, OUTW bit size results in excess bits that's regarded by synthesizer 
logic signed [INW*2-1:0] product, q;
logic input_valid_d; // delayed by one clk

assign product = input0 * input1; // first multiplier

    always_ff @(posedge clk) begin // this block updates q
        if(reset) begin
            q <= 0;
        end else begin
            q <= product;
        end
    end

    always_ff @(posedge clk) begin // this block updates input_valid_d
        if(reset) begin
            input_valid_d <= 0;
        end else begin
            input_valid_d <= input_valid;
        end
    end

    always_ff @(posedge clk) begin // this block updates output
        if(reset) begin
            out <= 0; // synch reset
        end else if(init_acc) begin
            out <= init_value;
        end else if(input_valid_d) begin
            out <= q + out;
        end else begin
            out <= out;
        end      
    end
endmodule
//--------------------------------------------------------------------------------------------------------------
//Part 2:
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
//--------------------------------------------------------------------------------------------------------------
//Part 3:
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

    memory #(.WIDTH(INW), .SIZE(MAXK*MAXK)) w_memory // instantiate memory module for w and m matrixes
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

    always_comb begin
        state_next = state;
        K_next = K_reg; // regs that will hold value of K and B
        B_next = B_reg;
        w_index_next = w_index_reg; // index registers
        x_index_next = x_index_reg;

        AXIS_TREADY = 0; // initialize singals
        inputs_loaded = 0;

        w_wr_en = 0;  // begin as not enabled
        x_wr_en = 0;

        w_addr = (state == DONE) ? W_read_addr : w_index_reg;   // selects which address source drives milder's memory modules (alliteration hehe)
        x_addr = (state == DONE) ? X_read_addr : x_index_reg;

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
                    end
                    else begin
                        x_wr_en = 1;        // if new_W = 0, just skip to load X
                        x_addr = 0; 
                        x_index_next = 1;
                        state_next = LOAD_X;
                    end
                end
            end

            LOAD_W: begin
                AXIS_TREADY = 1;        // set as ready
                if (valid_and_ready) begin
                    w_wr_en = 1;
                    w_addr = w_index_reg;

                    if (w_index_reg == last_w_index) begin   // We done after K*K addresses
                      
                        w_index_next = 0;                   // reset out w index
                        state_next = LOAD_B;
                    end
                    else begin
                        w_index_next = w_index_reg + 1;     // iterate
                    end
                end
            end

            LOAD_B: begin
                AXIS_TREADY = 1;        // set as ready
                if (valid_and_ready) begin
                    B_next = AXIS_TDATA;    // load B
                    x_index_next = 0;    // start X at address 0
                    state_next = LOAD_X;
                end
            end 

            LOAD_X: begin
                AXIS_TREADY = 1;        // set as ready
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
                AXIS_TREADY = 0;    // not ready to take more data
                inputs_loaded = 1;  // we done loading everything

                if (compute_finished) begin
                    state_next = IDLE; // go back to idle
                    w_index_next = 0; // reset indexes
                    x_index_next = 0; 
                end
            end

            default: begin
                state_next = IDLE;
            end
        endcase
    end

    always_ff @(posedge clk) begin  // update the states of everything
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

    // assign vals to outputs
    assign K = K_reg;
    assign B = B_reg;

endmodule
//--------------------------------------------------------------------------------------------------------------
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
