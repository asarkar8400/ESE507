//==============================================================================
// Part 5 (LEGAL MEMORY ONLY): Optimized Conv with N_MACS parallel MACs
// - Uses DesignWare pipelined multipliers (DW02_mult_*_stage) inside mac_pipe
// - N_MACS parallel MAC pipelines cooperate on ONE output window (Mode A)
// - LEGAL memory approach: replicate Part4 single-port memory N_MACS times
//   so each MAC has an independent read address
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
    logic signed [INW-1:0]     input0_reg, input1_reg;
    logic signed [2*INW-1:0]   product;
    logic [MULT_STAGES:0]      input_valid_pipe;

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
            out <= out + $signed(product); // sign-extend as needed
        end
    end

endmodule


//------------------------------------------------------------------------------
// Part 2: Simple dual-port memory with synchronous read (legal, from Part 4)
//------------------------------------------------------------------------------
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


//------------------------------------------------------------------------------
// Part 2: AXI-like FIFO for outputs (legal, from Part 4)
//------------------------------------------------------------------------------
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

    assign fifo_full  = (capacity == DEPTH);
    assign fifo_empty = (capacity == 0);

    assign OUT_AXIS_TVALID = !fifo_empty;
    assign IN_AXIS_TREADY  = (!fifo_full || (fifo_full && rd_en));

    assign wr_en = IN_AXIS_TVALID && IN_AXIS_TREADY;
    assign rd_en = OUT_AXIS_TVALID && OUT_AXIS_TREADY;

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

    memory_dual_port #(
        .SIZE (DEPTH),
        .WIDTH(OUTW)
    ) fifo_instance (
        .data_in    (IN_AXIS_TDATA),
        .data_out   (OUT_AXIS_TDATA),
        .write_addr (wr_ptr),
        .read_addr  (mem_read_addr),
        .clk        (clk),
        .wr_en      (wr_en)
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


//------------------------------------------------------------------------------
// Part 3: Single-port synchronous memory (legal, from Part 4)
//------------------------------------------------------------------------------
module memory #(
    parameter                   WIDTH=16,
    parameter                   SIZE=64,
    localparam                  LOGSIZE=$clog2(SIZE)
)(
    input  [WIDTH-1:0]          data_in,
    output logic [WIDTH-1:0]    data_out,
    input  [LOGSIZE-1:0]        addr,
    input                       clk,
    input                       wr_en
);

    logic [SIZE-1:0][WIDTH-1:0] mem;

    always_ff @(posedge clk) begin
        data_out <= mem[addr];
        if (wr_en)
            mem[addr] <= data_in;
    end
endmodule


//------------------------------------------------------------------------------
// Part 5 LEGAL Input memories:
// - Replicate W and X memories N_MACS times (LEGAL) to emulate N read ports
// - Broadcast writes to all replicas during load
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
    input  signed [INW-1:0]          AXIS_TDATA,
    input  logic                     AXIS_TVALID,
    input  [K_BITS:0]                AXIS_TUSER,   // [K_BITS:1]=K, [0]=new_W
    output logic                     AXIS_TREADY,

    output logic                     inputs_loaded,
    input  logic                     compute_finished,

    output logic [K_BITS-1:0]        K,
    output logic signed [INW-1:0]    B,

    input  [X_ADDR_BITS-1:0]         X_read_addr [N_MACS-1:0],
    output logic signed [INW-1:0]    X_data      [N_MACS-1:0],
    input  [W_ADDR_BITS-1:0]         W_read_addr [N_MACS-1:0],
    output logic signed [INW-1:0]    W_data      [N_MACS-1:0]
);

    // Decode TUSER
    logic [K_BITS-1:0] TUSER_K;
    logic              new_W;
    assign TUSER_K = AXIS_TUSER[K_BITS:1];
    assign new_W   = AXIS_TUSER[0];

    // FSM
    typedef enum logic [2:0] {IDLE, LOAD_W, LOAD_B, LOAD_X, DONE} load_state_t;
    load_state_t state, state_next;

    logic [K_BITS-1:0]       K_reg, K_next;
    logic signed [INW-1:0]   B_reg, B_next;

    logic [W_ADDR_BITS-1:0]  w_index_reg, w_index_next;
    logic [X_ADDR_BITS-1:0]  x_index_reg, x_index_next;

    localparam int X_LAST = R*C - 1;
    logic [W_ADDR_BITS-1:0] last_w_index;
    assign last_w_index = (K_reg * K_reg) - 1;

    logic valid_and_ready;
    assign valid_and_ready = AXIS_TVALID && AXIS_TREADY;

    // Write controls
    logic w_wr_en, x_wr_en;
    logic [W_ADDR_BITS-1:0] w_addr;
    logic [X_ADDR_BITS-1:0] x_addr;

    // Replicated memory outputs (unsigned wires, then cast to signed)
    logic [INW-1:0] w_data_u [N_MACS-1:0];
    logic [INW-1:0] x_data_u [N_MACS-1:0];

    // Address mux per replica: if writing, use loader addr; else use per-MAC read addr
    generate
        genvar m;
        for (m = 0; m < N_MACS; m++) begin : REPL_MEMS
            memory #(.WIDTH(INW), .SIZE(MAXK*MAXK)) w_memory (
                .clk     (clk),
                .data_in (AXIS_TDATA),
                .data_out(w_data_u[m]),
                .addr    (w_wr_en ? w_addr : W_read_addr[m]),
                .wr_en   (w_wr_en)
            );

            memory #(.WIDTH(INW), .SIZE(R*C)) x_memory (
                .clk     (clk),
                .data_in (AXIS_TDATA),
                .data_out(x_data_u[m]),
                .addr    (x_wr_en ? x_addr : X_read_addr[m]),
                .wr_en   (x_wr_en)
            );

            // Cast to signed for consumers
            always_comb begin
                W_data[m] = $signed(w_data_u[m]);
                X_data[m] = $signed(x_data_u[m]);
            end
        end
    endgenerate

    // FSM combinational
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

    // FSM sequential
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
// Part 5 Conv (Mode A): N_MACS cooperate on one output window
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

    localparam FIFO_DEPTH    = C - 1;
    localparam X_ADDR_BITS   = $clog2(R * C);
    localparam W_ADDR_BITS   = $clog2(MAXK * MAXK);
    localparam R_BITS        = $clog2(R);
    localparam C_BITS        = $clog2(C);
    localparam K_INDEX_BITS  = $clog2(MAXK);
    localparam MAC_LATENCY   = MULT_STAGES + 1;  // safe drain bound
    localparam MAC_WAIT_BITS = $clog2(MAC_LATENCY + 2);
    localparam MAC_INDEX_BITS= $clog2(MAXK*MAXK + 1);

    // Handshake with input_mems
    logic                    inputs_loaded;
    logic                    compute_finished;

    logic [K_BITS-1:0]       K;
    logic signed [INW-1:0]   B;

    // Per-MAC memory ports
    logic [X_ADDR_BITS-1:0]  X_read_addr [N_MACS-1:0];
    logic signed [INW-1:0]   X_data      [N_MACS-1:0];
    logic [W_ADDR_BITS-1:0]  W_read_addr [N_MACS-1:0];
    logic signed [INW-1:0]   W_data      [N_MACS-1:0];

    // Output FIFO interface
    logic [OUTW-1:0]         fifo_in_data;
    logic                    fifo_in_valid;
    logic                    fifo_in_ready;

    // MAC interfaces
    logic                    mac_init_acc;
    logic                    mac_input_valid [N_MACS-1:0];
    logic signed [OUTW-1:0]  mac_out [N_MACS-1:0];

    // Instantiate legal replicated memories
    input_mems #(
        .INW    (INW),
        .R      (R),
        .C      (C),
        .MAXK   (MAXK),
        .N_MACS (N_MACS)
    ) input_memory (
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

    fifo_out #(
        .OUTW  (OUTW),
        .DEPTH (FIFO_DEPTH)
    ) output_fifo (
        .clk            (clk),
        .reset          (reset),
        .IN_AXIS_TDATA  (fifo_in_data),
        .IN_AXIS_TVALID (fifo_in_valid),
        .IN_AXIS_TREADY (fifo_in_ready),
        .OUT_AXIS_TDATA (OUTPUT_TDATA),
        .OUT_AXIS_TVALID(OUTPUT_TVALID),
        .OUT_AXIS_TREADY(OUTPUT_TREADY)
    );

    // Bias into MAC[0], zeros into others
    logic signed [INW-1:0] mac_init_values [N_MACS-1:0];
    always_comb begin
        mac_init_values[0] = B;
        for (int m = 1; m < N_MACS; m++)
            mac_init_values[m] = '0;
    end

    // Instantiate N_MACS MAC pipelines
    genvar g;
    generate
        for (g = 0; g < N_MACS; g++) begin : MACS
            mac_pipe #(
                .INW         (INW),
                .OUTW        (OUTW),
                .MULT_STAGES (MULT_STAGES)
            ) mac_i (
                .clk        (clk),
                .reset      (reset),
                .input0     (X_data[g]),
                .input1     (W_data[g]),
                .init_value (mac_init_values[g]),
                .init_acc   (mac_init_acc),
                .input_valid(mac_input_valid[g]),
                .out        (mac_out[g])
            );
        end
    endgenerate

    // Sum MAC outputs (Mode A reduction)
    logic signed [OUTW-1:0] mac_sum;
    always_comb begin
        mac_sum = mac_out[0];
        for (int m = 1; m < N_MACS; m++)
            mac_sum += mac_out[m];
    end
    assign fifo_in_data = mac_sum;

    // FSM
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

    // Iteration + drain
    logic [MAC_INDEX_BITS-1:0] mac_iter, mac_iter_next;
    logic [MAC_WAIT_BITS-1:0]  wait_count, wait_count_next;

    // Rout/Cout
    logic [R_BITS-1:0] Rout;
    logic [C_BITS-1:0] Cout;
    assign Rout = R - K + 1;
    assign Cout = C - K + 1;

    // total ops
    logic [MAC_INDEX_BITS-1:0] total_ops;
    assign total_ops = K * K;

    // ceil(total_ops / N_MACS)
    logic [MAC_INDEX_BITS+3:0] total_ops_padded;
    logic [MAC_INDEX_BITS-1:0] num_iterations;
    assign total_ops_padded = {4'b0, total_ops} + (N_MACS - 1);
    assign num_iterations   = total_ops_padded / N_MACS;

    // Precompute small multiples of K (used in index advance)
    logic [K_BITS+1:0] K_x2, K_x3;
    assign K_x2 = K << 1;
    assign K_x3 = K + (K << 1);

    logic [K_INDEX_BITS+1:0] new_j_temp [N_MACS-1:0];
    logic [K_INDEX_BITS:0]   next_i_temp[N_MACS-1:0];
    logic [K_INDEX_BITS-1:0] next_j_temp[N_MACS-1:0];

    // Address generation
    always_comb begin
        for (int m = 0; m < N_MACS; m++) begin
            X_read_addr[m] = (r_index + i_index[m]) * C + (c_index + j_index[m]);
            W_read_addr[m] = i_index[m] * K + j_index[m];
        end
    end

    // Main FSM combinational
    always_comb begin
        state_next       = state;
        r_index_next     = r_index;
        c_index_next     = c_index;
        mac_iter_next    = mac_iter;
        wait_count_next  = wait_count;

        mac_init_acc     = 1'b0;
        fifo_in_valid    = 1'b0;
        compute_finished = 1'b0;

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
                mac_init_acc  = 1'b1;
                mac_iter_next = '0;

                // Initialize each MAC to kernel positions m
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
                // allow one cycle for memory synchronous read
                state_next = S_COMPUTE;
            end

            S_COMPUTE: begin
                // Fire valids for MACs that still have work this iteration
                for (int m = 0; m < N_MACS; m++) begin
                    if ((mac_iter * N_MACS + m) < total_ops)
                        mac_input_valid[m] = 1'b1;
                end

                mac_iter_next = mac_iter + 1;

                // Advance each MAC's (i,j) by +N_MACS positions without division
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

                if (mac_iter >= num_iterations - 1) begin
                    state_next      = S_WAIT;
                    wait_count_next = '0;
                end
            end

            S_WAIT: begin
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
            end

            default: state_next = S_IDLE;
        endcase
    end

    // FSM sequential
    always_ff @(posedge clk) begin
        if (reset) begin
            state      <= S_IDLE;
            r_index    <= '0;
            c_index    <= '0;
            mac_iter   <= '0;
            wait_count <= '0;
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
            for (int m=0; m<N_MACS; m++) begin
                i_index[m] <= i_index_next[m];
                j_index[m] <= j_index_next[m];
            end
        end
    end

endmodule
