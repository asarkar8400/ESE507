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
