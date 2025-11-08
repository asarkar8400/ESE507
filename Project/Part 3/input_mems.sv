// Memory to use for input_memory module
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

// input_mems module
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
    
    input compute_finished,
    input [X_ADDR_BITS-1:0] X_read_addr,
    input [W_ADDR_BITS-1:0] W_read_addr,

    output logic [K_BITS-1:0] K,
    output logic signed [INW-1:0] B,
    output logic signed [INW-1:0] X_data,
    output logic signed [INW-1:0] W_data,
    output logic inputs_loaded
    
);

    // TUSER_K assignments
    logic [$clog2(MAXK+1)-1:0] TUSER_K; // register that provides value of K
    assign TUSER_K = AXIS_TUSER[$clog2(MAXK+1):1]; 
   
    logic new_W;                        // indicates if there is a new W matrix
    assign new_W = AXIS_TUSER[0];

    // logic needed for memory module instantiations
    logic w_wr_en, x_wr_en;
    logic signed [INW-1:0] w_data_out, x_data_out;
    logic [W_ADDR_BITS-1:0] w_addr;
    logic [X_ADDR_BITS-1:0] x_addr;
    logic [W_ADDR_BITS-1:0] w_count_curr, w_count_next;
    logic [X_ADDR_BITS-1:0] x_count_curr, x_count_next;

    logic [K_BITS-1:0] K_curr, K_next;
    logic signed [INW-1:0] B_curr, B_next;


    logic [W_ADDR_BITS-1:0] KxK; // K x K elements; Used as max value to iterate to when loading W matrix
    assign KxK = K_curr * K_curr;

    // W matrix memory instantiation
    memory #(.WIDTH(INW), .SIZE(MAXK * MAXK)) 
    w_memory (
        .clk(clk),
        .data_in(AXIS_TDATA),
        .data_out(w_data_out),
        .addr(w_addr),
        .wr_en(w_wr_en)
    );

    // X matrix memory instantiation
    memory #(.WIDTH(INW), .SIZE(R * C)) 
    x_memory (
        .clk(clk),
        .data_in(AXIS_TDATA),
        .data_out(x_data_out),
        .addr(x_addr),
        .wr_en(x_wr_en)
    );

     // State assignment using enums:
    enum logic [2:0] {INITIAL, LOADW, LOADB, LOADX, DONE} state, next_state;

    // state transitions logic
    always_comb begin
        // initial state of signals
        next_state = state; // the default
        
        K_next = K_curr;
        B_next = B_curr;
        w_count_next = w_count_curr;
        x_count_next = x_count_curr;
        
        AXIS_TREADY = 0;
        inputs_loaded = 0;
        
        w_wr_en = 0;
        x_wr_en = 0;
        w_addr = '0;
        x_addr = '0;

        case (state)
            INITIAL: begin
                AXIS_TREADY = 1;
                if (AXIS_TVALID && AXIS_TREADY) begin // if valid and ready proceed
                    if(new_W) begin
                        // load value of K -> Load W[0,0] (Waveform in PDF reads W[0,0] right when K gets loaded) -> Go to LOADW state
                        K_next = TUSER_K;
                        
                        w_wr_en = 1;
                        w_addr = 0;         // Load W[0,0] 
                        w_count_next = 1;

                        next_state = LOADW; // Go to LOADW state

                    end else begin
                        // Skip LoadW and LoadB, Load X[0,0] -> Go to LOADX state
                        x_wr_en = 1;
                        x_addr = 0;         // Load X[0,0]
                        x_count_next = 1;
                        next_state = LOADX; // Go to LOADX state
                    end
                end
            end

            LOADW: begin
                AXIS_TREADY = 1;
                if (AXIS_TVALID && AXIS_TREADY) begin    // if valid and ready proceed
                    // Iterate w_counter to go load each element -> Load W[i] -> Go to LOADB state
                    w_wr_en = 1;
                    w_addr = w_count_curr;               // Load W[i]
                    w_count_next = w_count_curr + 1;     // iterate w_count_next (i)
                    if(w_count_curr == KxK - 1) begin    // if counter reaches last index, go to next state
                        w_count_next = 0;                // reset counter
                        next_state = LOADB;              // Go to LOADW state
                    end
                end
            end

            LOADB: begin
                AXIS_TREADY = 1;
                if (AXIS_TVALID && AXIS_TREADY) begin // if valid and ready proceed
                    // Load B
                    B_next = AXIS_TDATA;
                    next_state = LOADX;
                end
            end

            LOADX: begin
                AXIS_TREADY = 1;
                if (AXIS_TVALID && AXIS_TREADY) begin // if valid and ready proceed
                    // Iterate w_counter to go load each element -> Load W[0,0] (Waveform in PDF reads W[0,0] right when K gets loaded) -> Go to LOADW state
                    x_wr_en = 1;
                    x_addr = x_count_curr; // Load X[j]
                    x_count_next = x_count_curr + 1;        // iterate x_count_next (j)
                    if(x_count_curr == ((R*C) - 1)) begin   // if counter reaches last index, go to next state
                        x_count_next = 0;                   // reset counter
                        next_state = DONE;                 // Go to DONE state
                    end
                end
            end

            DONE: begin
                AXIS_TREADY = 0;
                inputs_loaded = 1;
                x_addr = X_read_addr; //read input memories
                w_addr = W_read_addr; 

                if(compute_finished) begin
                    next_state = INITIAL;
                end
            end

            default: begin
                next_state = INITIAL;
            end
        endcase
    end

    // state registers
    always_ff @(posedge clk) begin
        if (reset) begin
            state <= INITIAL;
            K_curr <= '0;
            B_curr <= '0;
            w_count_curr <= '0;
            x_count_curr <= '0;
        end else begin
            state <= next_state;
            K_curr <= K_next;
            B_curr <= B_next;
            w_count_curr <= w_count_next;
            x_count_curr <= x_count_next;
        end
    end
    
    
    assign X_data = x_data_out;
    assign W_data = w_data_out;
    assign K = K_curr;
    assign B = B_curr;

endmodule
