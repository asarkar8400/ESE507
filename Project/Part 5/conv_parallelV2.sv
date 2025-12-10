//==============================================================================
// Part 5: Optimized Conv Module with Parallel MAC Units
// - Uses DesignWare pipelined multipliers (DW02_mult_*_stage)
// - N_MACS parallel MAC pipelines
// - No runtime division in main kernel scheduling
//==============================================================================

//------------------------------------------------------------------------------
// Part 1: MAC with Pipelined DesignWare Multiplier
//------------------------------------------------------------------------------
module mac_pipe #(
    parameter INW         = 16, 
    parameter OUTW        = 64,
    parameter MULT_STAGES = 6   // 2..6 for DW02
)(
    input  logic                      clk,
    input  logic                      reset,
    input  signed [INW-1:0]           input0,
    input  signed [INW-1:0]           input1,
    input  signed [INW-1:0]           init_value,
    input  logic                      init_acc,
    input  logic                      input_valid,
    output logic signed [OUTW-1:0]    out
);

    // Register inputs to multiplier
    logic signed [INW-1:0] input0_reg, input1_reg;
    logic signed [2*INW-1:0] product;
    logic [MULT_STAGES:0]    input_valid_pipe;

    // Input register
    always_ff @(posedge clk) begin
        if (reset) begin
            input0_reg <= '0;
            input1_reg <= '0;
        end else begin
            input0_reg <= input0;
            input1_reg <= input1;
        end
    end

    // DesignWare pipelined multipliers
    generate
        if (MULT_STAGES == 2) begin : g_mult2
            DW02_mult_2_stage #(.A_width(INW), .B_width(INW)) mult_i (
                .A      (input0_reg),
                .B      (input1_reg),
                .TC     (1'b1),
                .CLK    (clk),
                .PRODUCT(product)
            );
        end else if (MULT_STAGES == 3) begin : g_mult3
            DW02_mult_3_stage #(.A_width(INW), .B_width(INW)) mult_i (
                .A      (input0_reg),
                .B      (input1_reg),
                .TC     (1'b1),
                .CLK    (clk),
                .PRODUCT(product)
            );
        end else if (MULT_STAGES == 4) begin : g_mult4
            DW02_mult_4_stage #(.A_width(INW), .B_width(INW)) mult_i (
                .A      (input0_reg),
                .B      (input1_reg),
                .TC     (1'b1),
                .CLK    (clk),
                .PRODUCT(product)
            );
        end else if (MULT_STAGES == 5) begin : g_mult5
            DW02_mult_5_stage #(.A_width(INW), .B_width(INW)) mult_i (
                .A      (input0_reg),
                .B      (input1_reg),
                .TC     (1'b1),
                .CLK    (clk),
                .PRODUCT(product)
            );
        end else begin : g_mult6 // default MULT_STAGES=6
            DW02_mult_6_stage #(.A_width(INW), .B_width(INW)) mult_i (
                .A      (input0_reg),
                .B      (input1_reg),
                .TC     (1'b1),
                .CLK    (clk),
                .PRODUCT(product)
            );
        end
    endgenerate

    // Pipeline the valid bit to match multiplier latency
    always_ff @(posedge clk) begin
        if (reset) begin
            input_valid_pipe <= '0;
        end else begin
            input_valid_pipe <= {input_valid_pipe[MULT_STAGES-1:0], input_valid};
        end
    end

    // Accumulator
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

//------------------------------------------------------------------------------
// Simple dual-port memory with synchronous read
//------------------------------------------------------------------------------
module memory_dual_port #(
    parameter WIDTH   = 24,
    parameter SIZE    = 19,
    localparam LOGSZ  = $clog2(SIZE)
)(
    input  logic                  clk,
    input  logic                  wr_en,
    input  [WIDTH-1:0]            data_in,
    input  [LOGSZ-1:0]            write_addr,
    input  [LOGSZ-1:0]            read_addr,
    output logic [WIDTH-1:0]      data_out
);
    logic [SIZE-1:0][WIDTH-1:0] mem = '{default:'0};

    always_ff @(posedge clk) begin
        // Read
        if (wr_en && (read_addr == write_addr))
            data_out <= data_in;
        else
            data_out <= mem[read_addr];

        // Write
        if (wr_en)
            mem[write_addr] <= data_in;
    end
endmodule

//------------------------------------------------------------------------------
// AXI-like FIFO for outputs
//------------------------------------------------------------------------------
module fifo_out #(
    parameter OUTW   = 24,
    parameter DEPTH  = 19,
    localparam LOGD  = $clog2(DEPTH)
)(
    input  logic                 clk,
    input  logic                 reset,
    // input side
    input  [OUTW-1:0]            IN_AXIS_TDATA,
    input  logic                 IN_AXIS_TVALID,
    output logic                 IN_AXIS_TREADY,
    // output side
    output logic [OUTW-1:0]      OUT_AXIS_TDATA,
    output logic                 OUT_AXIS_TVALID,
    input  logic                 OUT_AXIS_TREADY
);

    logic [LOGD-1:0] wr_ptr, rd_ptr, rd_ptr_next;
    logic [LOGD:0]   capacity;
    logic            wr_en, rd_en;
    logic [LOGD-1:0] mem_read_addr;
    logic            fifo_full, fifo_empty;

    assign fifo_full      = (capacity == DEPTH);
    assign fifo_empty     = (capacity == 0);
    assign OUT_AXIS_TVALID = !fifo_empty;

    // Allow write if not full OR if we read in same cycle (classic FIFO trick)
    assign wr_en          = IN_AXIS_TVALID && IN_AXIS_TREADY;
    assign rd_en          = OUT_AXIS_TVALID && OUT_AXIS_TREADY;
    assign IN_AXIS_TREADY = (!fifo_full) || (fifo_full && rd_en);

    // Next read pointer
    always_comb begin
        if (rd_en) begin
            if (rd_ptr == DEPTH-1)
                rd_ptr_next = '0;
            else
                rd_ptr_next = rd_ptr + 1;
        end else begin
            rd_ptr_next = rd_ptr;
        end
    end

    assign mem_read_addr = rd_ptr_next;

    // Underlying memory
    memory_dual_port #(
        .WIDTH (OUTW),
        .SIZE  (DEPTH)
    ) fifo_mem (
        .clk        (clk),
        .wr_en      (wr_en),
        .data_in    (IN_AXIS_TDATA),
        .write_addr (wr_ptr),
        .read_addr  (mem_read_addr),
        .data_out   (OUT_AXIS_TDATA)
    );

    // Pointers and occupancy
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

//------------------------------------------------------------------------------
// Multi-port memory for X/W: N_PORTS independent read addresses
//------------------------------------------------------------------------------
module memory_multi_port #(   
    parameter WIDTH    = 16, 
    parameter SIZE     = 64,
    parameter N_PORTS  = 4,
    localparam LOGSZ   = $clog2(SIZE)
)(
    input  logic                       clk,
    input  logic                       wr_en,
    input  signed [WIDTH-1:0]          data_in,
    input  [LOGSZ-1:0]                 write_addr,
    input  [LOGSZ-1:0]                 read_addr [N_PORTS-1:0],
    output logic signed [WIDTH-1:0]    data_out [N_PORTS-1:0]
);
    logic signed [SIZE-1:0][WIDTH-1:0] mem = '{default:'0};

    always_ff @(posedge clk) begin
        for (int i = 0; i < N_PORTS; i++) begin
            data_out[i] <= mem[read_addr[i]];
        end
        if (wr_en)
            mem[write_addr] <= data_in;
    end
endmodule

//------------------------------------------------------------------------------
// Input memories: load W, B, X once; then support N_MACS parallel reads
//------------------------------------------------------------------------------
module input_mems #(
    parameter INW    = 24,
    parameter R      = 9,
    parameter C      = 8,
    parameter MAXK   = 4,
    parameter N_MACS = 4,
    localparam K_BITS      = $clog2(MAXK+1),
    localparam X_ADDR_BITS = $clog2(R*C),
    localparam W_ADDR_BITS = $clog2(MAXK*MAXK)
)(
    input  logic                     clk,
    input  logic                     reset,
    // AXI-like input
    input  signed [INW-1:0]          AXIS_TDATA,
    input  logic                     AXIS_TVALID,
    input  [K_BITS:0]                AXIS_TUSER,   // [K_BITS:1] = K, [0]=new_W
    output logic                     AXIS_TREADY,
    // handshaking with Conv
    output logic                     inputs_loaded,
    input  logic                     compute_finished,
    // outputs
    output logic [K_BITS-1:0]        K,
    output logic signed [INW-1:0]    B,
    // addresses from Conv
    input  [X_ADDR_BITS-1:0]         X_read_addr [N_MACS-1:0],
    output logic signed [INW-1:0]    X_data      [N_MACS-1:0],
    input  [W_ADDR_BITS-1:0]         W_read_addr [N_MACS-1:0],
    output logic signed [INW-1:0]    W_data      [N_MACS-1:0]
);

    // Decode AXIS_TUSER
    logic [K_BITS-1:0] TUSER_K;
    logic              new_W;
    assign TUSER_K = AXIS_TUSER[K_BITS:1];
    assign new_W   = AXIS_TUSER[0];

    // Memory connections
    logic signed [INW-1:0] w_data_out [N_MACS-1:0];
    logic signed [INW-1:0] x_data_out [N_MACS-1:0];

    logic [W_ADDR_BITS-1:0] w_addr;
    logic [X_ADDR_BITS-1:0] x_addr;
    logic                   w_wr_en, x_wr_en;

    logic [W_ADDR_BITS-1:0] w_read_addr_array [N_MACS-1:0];
    logic [X_ADDR_BITS-1:0] x_read_addr_array [N_MACS-1:0];

    // Instantiate W/X memories
    memory_multi_port #(
        .WIDTH   (INW),
        .SIZE    (MAXK*MAXK),
        .N_PORTS (N_MACS)
    ) w_mem (
        .clk      (clk),
        .wr_en    (w_wr_en),
        .data_in  (AXIS_TDATA),
        .write_addr(w_addr),
        .read_addr(w_read_addr_array),
        .data_out (w_data_out)
    );

    memory_multi_port #(
        .WIDTH   (INW),
        .SIZE    (R*C),
        .N_PORTS (N_MACS)
    ) x_mem (
        .clk      (clk),
        .wr_en    (x_wr_en),
        .data_in  (AXIS_TDATA),
        .write_addr(x_addr),
        .read_addr(x_read_addr_array),
        .data_out (x_data_out)
    );

    always_comb begin
        for (int i = 0; i < N_MACS; i++) begin
            w_read_addr_array[i] = W_read_addr[i];
            x_read_addr_array[i] = X_read_addr[i];
        end
    end

    assign W_data = w_data_out;
    assign X_data = x_data_out;

    // Load FSM
    typedef enum logic [2:0] {IDLE, LOAD_W, LOAD_B, LOAD_X, DONE} load_state_t;
    load_state_t state, state_next;

    logic [K_BITS-1:0]       K_reg, K_next;
    logic signed [INW-1:0]   B_reg, B_next;
    logic [W_ADDR_BITS-1:0]  w_index_reg, w_index_next;
    logic [X_ADDR_BITS-1:0]  x_index_reg, x_index_next;

    localparam int X_LAST = R*C - 1;
    logic [W_ADDR_BITS-1:0]  last_w_index;
    assign last_w_index = (K_reg * K_reg) - 1;

    logic valid_and_ready;
    assign valid_and_ready = AXIS_TVALID && AXIS_TREADY;

    always_comb begin
        state_next    = state;
        K_next        = K_reg;
        B_next        = B_reg;
        w_index_next  = w_index_reg;
        x_index_next  = x_index_reg;

        AXIS_TREADY   = 1'b0;
        inputs_loaded = 1'b0;
        w_wr_en       = 1'b0;
        x_wr_en       = 1'b0;
        w_addr        = w_index_reg;
        x_addr        = x_index_reg;

        case (state)
            IDLE: begin
                AXIS_TREADY = 1'b1;
                if (valid_and_ready) begin
                    if (new_W) begin
                        K_next       = TUSER_K;
                        w_wr_en      = 1'b1;
                        w_addr       = '0;
                        w_index_next = 1;
                        x_index_next = '0;
                        state_next   = LOAD_W;
                    end else begin
                        // Re-use old W/B, only load X
                        x_wr_en      = 1'b1;
                        x_addr       = '0;
                        x_index_next = 1;
                        state_next   = LOAD_X;
                    end
                end
            end

            LOAD_W: begin
                AXIS_TREADY = 1'b1;
                if (valid_and_ready) begin
                    w_wr_en = 1'b1;
                    w_addr  = w_index_reg;
                    if (w_index_reg == last_w_index) begin
                        w_index_next = '0;
                        state_next   = LOAD_B;
                    end else begin
                        w_index_next = w_index_reg + 1;
                    end
                end
            end

            LOAD_B: begin
                AXIS_TREADY = 1'b1;
                if (valid_and_ready) begin
                    B_next       = AXIS_TDATA;
                    x_index_next = '0;
                    state_next   = LOAD_X;
                end
            end

            LOAD_X: begin
                AXIS_TREADY = 1'b1;
                if (valid_and_ready) begin
                    x_wr_en = 1'b1;
                    x_addr  = x_index_reg;
                    if (x_index_reg == X_LAST[X_ADDR_BITS-1:0]) begin
                        x_index_next = '0;
                        state_next   = DONE;
                    end else begin
                        x_index_next = x_index_reg + 1;
                    end
                end
            end

            DONE: begin
                AXIS_TREADY   = 1'b0;
                inputs_loaded = 1'b1;
                if (compute_finished) begin
                    state_next   = IDLE;
                    w_index_next = '0;
                    x_index_next = '0;
                end
            end

            default: state_next = IDLE;
        endcase
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            state       <= IDLE;
            K_reg       <= '0;
            B_reg       <= '0;
            w_index_reg <= '0;
            x_index_reg <= '0;
        end else begin
            state       <= state_next;
            K_reg       <= K_next;
            B_reg       <= B_next;
            w_index_reg <= w_index_next;
            x_index_reg <= x_index_next;
        end
    end

    assign K = K_reg;
    assign B = B_reg;

endmodule
//------------------------------------------------------------------------------
// Conv with N_MACS parallel MAC units
// Schedules K*K multiplies over N_MACS engines using index arithmetic only
//------------------------------------------------------------------------------
module Conv #(
    parameter INW         = 18,
    parameter R           = 8,
    parameter C           = 8,
    parameter MAXK        = 5,
    parameter MULT_STAGES = 6,
    parameter N_MACS      = 4,
    localparam OUTW       = $clog2(MAXK*MAXK*(128'd1 << (2*INW-2)) + (1<<(INW-1)))+1,
    localparam K_BITS     = $clog2(MAXK+1)
)(
    input  logic                    clk, 
    input  logic                    reset,
    input  signed [INW-1:0]         INPUT_TDATA,
    input  logic                    INPUT_TVALID,
    input  [K_BITS:0]               INPUT_TUSER, 
    output logic                    INPUT_TREADY, 
    output logic [OUTW-1:0]         OUTPUT_TDATA,
    output logic                    OUTPUT_TVALID,
    input  logic                    OUTPUT_TREADY
);

    // For debugging / checking that N_MACS is what you think
    initial begin
        $display("========================================");
        $display("Conv instantiated: N_MACS = %0d, MULT_STAGES = %0d", N_MACS, MULT_STAGES);
        $display("========================================");
    end

    localparam FIFO_DEPTH    = C - 1;
    localparam X_ADDR_BITS   = $clog2(R * C);
    localparam W_ADDR_BITS   = $clog2(MAXK * MAXK);
    localparam R_BITS        = $clog2(R);
    localparam C_BITS        = $clog2(C);
    localparam K_INDEX_BITS  = $clog2(MAXK);
    localparam MAC_LATENCY   = MULT_STAGES + 1;  // multiplier + acc alignment
    localparam MAC_WAIT_BITS = $clog2(MAC_LATENCY + 2);
    localparam MAC_INDEX_BITS= $clog2(MAXK*MAXK + 1);

    // Handshake with input_mems
    logic                    inputs_loaded;
    logic                    compute_finished;

    logic [K_BITS-1:0]       K;
    logic signed [INW-1:0]   B;

    // Memory read ports
    logic [X_ADDR_BITS-1:0]  X_read_addr [N_MACS-1:0];
    logic signed [INW-1:0]   X_data      [N_MACS-1:0];
    logic [W_ADDR_BITS-1:0]  W_read_addr [N_MACS-1:0];
    logic signed [INW-1:0]   W_data      [N_MACS-1:0];

    // FIFO
    logic [OUTW-1:0]         fifo_in_data;
    logic                    fifo_in_valid;
    logic                    fifo_in_ready;

    // MAC interfaces
    logic signed [INW-1:0]   mac_input0 [N_MACS-1:0];
    logic signed [INW-1:0]   mac_input1 [N_MACS-1:0];
    logic                    mac_init_acc;
    logic                    mac_input_valid [N_MACS-1:0];
    logic signed [OUTW-1:0]  mac_out [N_MACS-1:0];

    // Instantiate memories
    input_mems #(
        .INW    (INW),
        .R      (R),
        .C      (C),
        .MAXK   (MAXK),
        .N_MACS (N_MACS)
    ) inp (
        .clk            (clk),
        .reset          (reset),
        .AXIS_TDATA     (INPUT_TDATA),
        .AXIS_TVALID    (INPUT_TVALID),
        .AXIS_TUSER     (INPUT_TUSER),
        .AXIS_TREADY    (INPUT_TREADY),
        .inputs_loaded  (inputs_loaded),
        .compute_finished(compute_finished),
        .K              (K),
        .B              (B),
        .X_read_addr    (X_read_addr),
        .X_data         (X_data),
        .W_read_addr    (W_read_addr),
        .W_data         (W_data)
    );

    fifo_out #(
        .OUTW  (OUTW),
        .DEPTH (FIFO_DEPTH)
    ) out_fifo (
        .clk            (clk),
        .reset          (reset),
        .IN_AXIS_TDATA  (fifo_in_data),
        .IN_AXIS_TVALID (fifo_in_valid),
        .IN_AXIS_TREADY (fifo_in_ready),
        .OUT_AXIS_TDATA (OUTPUT_TDATA),
        .OUT_AXIS_TVALID(OUTPUT_TVALID),
        .OUT_AXIS_TREADY(OUTPUT_TREADY)
    );

    // Bias into MAC[0], zero into others
    logic signed [INW-1:0] mac_init_values [N_MACS-1:0];
    always_comb begin
        mac_init_values[0] = B;
        for (int m = 1; m < N_MACS; m++) begin
            mac_init_values[m] = '0;
        end
    end

    // Instantiate N_MACS MAC pipelines
    genvar g;
    generate
        for (g = 0; g < N_MACS; g++) begin : g_macs
            mac_pipe #(
                .INW         (INW),
                .OUTW        (OUTW),
                .MULT_STAGES (MULT_STAGES)
            ) mac_i (
                .clk        (clk),
                .reset      (reset),
                .input0     (mac_input0[g]),
                .input1     (mac_input1[g]),
                .init_value (mac_init_values[g]),
                .init_acc   (mac_init_acc),
                .input_valid(mac_input_valid[g]),
                .out        (mac_out[g])
            );
        end
    endgenerate

    // Sum of all MAC outputs (they are sharing the same accumulator state)
    logic signed [OUTW-1:0] mac_sum;
    always_comb begin
        mac_sum = mac_out[0];
        for (int m = 1; m < N_MACS; m++) begin
            mac_sum = mac_sum + mac_out[m];
        end
    end
    assign fifo_in_data = mac_sum;

    // FSM for convolution
    typedef enum logic [2:0] {S_IDLE, S_INIT, S_READ, S_COMPUTE, S_WAIT, S_WRITE, S_DONE} state_t;
    state_t state, state_next;

    // Output indices
    logic [R_BITS-1:0] r_index, r_index_next;
    logic [C_BITS-1:0] c_index, c_index_next;

    // Per-MAC kernel indices (i,j)
    logic [K_INDEX_BITS-1:0] i_index      [N_MACS-1:0];
    logic [K_INDEX_BITS-1:0] j_index      [N_MACS-1:0];
    logic [K_INDEX_BITS-1:0] i_index_next [N_MACS-1:0];
    logic [K_INDEX_BITS-1:0] j_index_next [N_MACS-1:0];

    // Iteration counter and wait counter
    logic [MAC_INDEX_BITS-1:0] mac_iter, mac_iter_next;
    logic [MAC_WAIT_BITS-1:0]  wait_count, wait_count_next;

    // Derived sizes
    logic [R_BITS-1:0] Rout;
    logic [C_BITS-1:0] Cout;
    assign Rout = R - K + 1;
    assign Cout = C - K + 1;

    // Total operations and iterations
    logic [MAC_INDEX_BITS-1:0] total_ops;
    assign total_ops = K * K;

    // ceil(K*K / N_MACS)
    logic [MAC_INDEX_BITS+3:0] total_ops_padded;
    logic [MAC_INDEX_BITS-1:0] num_iterations;
    assign total_ops_padded = {4'b0, total_ops} + (N_MACS - 1);
    assign num_iterations    = total_ops_padded / N_MACS;

    // Some precomputed multiples of K (for index advance)
    logic [K_BITS+1:0] K_x2, K_x3;
    assign K_x2 = K << 1;
    assign K_x3 = K + (K << 1);

    // Temps for index advance
    logic [K_INDEX_BITS+1:0] new_j_temp [N_MACS-1:0];
    logic [K_INDEX_BITS:0]   next_i_temp[N_MACS-1:0];
    logic [K_INDEX_BITS-1:0] next_j_temp[N_MACS-1:0];

    // Global cycle counter (for debug)
    logic [31:0] global_cycle_count;

    // Generate memory addresses and feed MAC inputs
    always_comb begin
        for (int m = 0; m < N_MACS; m++) begin
            X_read_addr[m] = (r_index + i_index[m]) * C + (c_index + j_index[m]);
            W_read_addr[m] = i_index[m] * K + j_index[m];
            mac_input0[m]  = X_data[m];
            mac_input1[m]  = W_data[m];
        end
    end

    // Main FSM
    always_comb begin
        state_next        = state;
        r_index_next      = r_index;
        c_index_next      = c_index;
        mac_iter_next     = mac_iter;
        wait_count_next   = wait_count;
        mac_init_acc      = 1'b0;
        fifo_in_valid     = 1'b0;
        compute_finished  = 1'b0;

        for (int m = 0; m < N_MACS; m++) begin
            mac_input_valid[m] = 1'b0;
            i_index_next[m]    = i_index[m];
            j_index_next[m]    = j_index[m];
            new_j_temp[m]      = '0;
            next_i_temp[m]     = '0;
            next_j_temp[m]     = '0;
        end

        case (state)

            S_IDLE: begin
                if (inputs_loaded) begin
                    r_index_next  = '0;
                    c_index_next  = '0;
                    mac_iter_next = '0;
                    state_next    = S_INIT;
                end
            end

            S_INIT: begin
                // Initialize accumulators
                mac_init_acc  = 1'b1;
                mac_iter_next = '0;

                // Initial (i,j) for each MAC: positions 0..N_MACS-1
                // Note: small m/K & m%K here; synthesizes to comparators
                for (int m = 0; m < N_MACS; m++) begin
                    if (m < total_ops) begin
                        i_index_next[m] = m / K;
                        j_index_next[m] = m % K;
                    end else begin
                        i_index_next[m] = '0;
                        j_index_next[m] = '0;
                    end
                end

                state_next = S_READ;
            end

            S_READ: begin
                // One cycle to get first X/W data from memory
                state_next = S_COMPUTE;
            end

            S_COMPUTE: begin
                // Assert input_valid for MACs that still have work
                for (int m = 0; m < N_MACS; m++) begin
                    if ((mac_iter * N_MACS + m) < total_ops)
                        mac_input_valid[m] = 1'b1;
                end

                mac_iter_next = mac_iter + 1;

                // Advance each MAC's (i,j) by N_MACS positions using only
                // addition, comparison, and subtraction (no runtime divide).
                for (int m = 0; m < N_MACS; m++) begin
                    new_j_temp[m] = j_index[m] + N_MACS;

                    if (new_j_temp[m] >= K_x3) begin
                        next_j_temp[m] = new_j_temp[m] - K_x3;
                        next_i_temp[m] = i_index[m] + 3;
                    end else if (new_j_temp[m] >= K_x2) begin
                        next_j_temp[m] = new_j_temp[m] - K_x2;
                        next_i_temp[m] = i_index[m] + 2;
                    end else if (new_j_temp[m] >= K) begin
                        next_j_temp[m] = new_j_temp[m] - K;
                        next_i_temp[m] = i_index[m] + 1;
                    end else begin
                        next_j_temp[m] = new_j_temp[m];
                        next_i_temp[m] = i_index[m];
                    end

                    i_index_next[m] = next_i_temp[m][K_INDEX_BITS-1:0];
                    j_index_next[m] = next_j_temp[m];
                end

                // Finished issuing all valid K*K products?
                if (mac_iter >= num_iterations - 1) begin
                    state_next      = S_WAIT;
                    wait_count_next = '0;
                end
            end

            S_WAIT: begin
                // Wait for MAC pipeline to drain
                if (wait_count >= MAC_LATENCY-1) begin
                    state_next = S_WRITE;
                end else begin
                    wait_count_next = wait_count + 1;
                end
            end

            S_WRITE: begin
                fifo_in_valid = 1'b1;
                if (fifo_in_ready) begin
                    if (c_index == (Cout-1)) begin
                        c_index_next = '0;
                        if (r_index == (Rout-1)) begin
                            state_next = S_DONE;
                        end else begin
                            r_index_next  = r_index + 1;
                            mac_iter_next = '0;
                            state_next    = S_INIT;
                        end
                    end else begin
                        c_index_next  = c_index + 1;
                        mac_iter_next = '0;
                        state_next    = S_INIT;
                    end
                end
            end

            S_DONE: begin
                compute_finished = 1'b1;
                state_next       = S_IDLE;

                // Debug stats for one full Conv
                $display("========================================");
                $display("Conv DONE: N_MACS=%0d, K=%0d", N_MACS, K);
                $display("  total_ops    = %0d (K*K)", total_ops);
                $display("  num_iterations (ceil) = %0d", num_iterations);
                $display("========================================");
            end

            default: state_next = S_IDLE;
        endcase
    end

    // Sequential part
    always_ff @(posedge clk) begin
        if (reset) begin
            state            <= S_IDLE;
            r_index          <= '0;
            c_index          <= '0;
            mac_iter         <= '0;
            wait_count       <= '0;
            global_cycle_count <= '0;
            for (int m=0; m<N_MACS; m++) begin
                i_index[m] <= '0;
                j_index[m] <= '0;
            end
        end else begin
            state      <= state_next;
            r_index    <= r_index_next;
            c_index    <= c_index_next;
            mac_iter   <= mac_iter_next;
            wait_count <= wait_count_next;

            // global counter (purely for debug, if you want)
            if (state != S_IDLE)
                global_cycle_count <= global_cycle_count + 1;

            for (int m=0; m<N_MACS; m++) begin
                i_index[m] <= i_index_next[m];
                j_index[m] <= j_index_next[m];
            end
        end
    end

endmodule
