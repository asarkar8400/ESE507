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
