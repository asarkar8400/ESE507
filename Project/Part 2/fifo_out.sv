// dual port memory module
module memory_dual_port #(
        parameter                WIDTH=16, SIZE=64,
        localparam               LOGSIZE=$clog2(SIZE)
    )(
        input [WIDTH-1:0]        data_in,
        output logic [WIDTH-1:0] data_out,
        input [LOGSIZE-1:0]      write_addr, read_addr,
        input                    clk, wr_en
    );

    logic [SIZE-1:0][WIDTH-1:0] mem;

    always_ff @(posedge clk) begin
        // if we are reading and writing to same address concurrently, 
        // then output the new data
        if (wr_en && (read_addr == write_addr))
            data_out <= data_in;
        else
            data_out <= mem[read_addr];

        if (wr_en)
            mem[write_addr] <= data_in;            
    end
endmodule
----------------------------------------------------------------------------
// axi fifo module
module fifo_out #(
    parameter OUTW = 24,
    parameter DEPTH = 19,
    localparam LOGDEPTH = $clog2(DEPTH)
    )
    (
    input clk,
    input reset,
    input [OUTW-1:0] IN_AXIS_TDATA,
    input IN_AXIS_TVALID,
    output logic IN_AXIS_TREADY,
    output logic [OUTW-1:0] OUT_AXIS_TDATA,
    output logic OUT_AXIS_TVALID,
    input OUT_AXIS_TREADY
    );

// read and write pointers
logic [LOGDEPTH -1:0] wr_ptr, rd_ptr;
// write enable
logic wr_en;

// instantiate the memory_dual_port module 
memory_dual_port #(.SIZE(DEPTH), .WIDTH(OUTW)) 
fifo_instance(.data_in(IN_AXIS_TDATA), .clk(clk), .wr_en(wr_en), .write_addr(wr_ptr), .read_addr(rd_ptr), .data_out(OUT_AXIS_TDATA));

logic [LOGDEPTH:0] capacity;
logic fifo_full, fifo_empty;
logic rd_en;

// fifo full and empty signals
assign fifo_full = (capacity == DEPTH);
assign fifo_empty = (capacity == 0);

// AXI Stream interface signals
assign OUT_AXIS_TVALID = !fifo_empty;
assign IN_AXIS_TREADY = (!fifo_full || fifo_full && rd_en);

// read and write enable logic
assign wr_en = IN_AXIS_TVALID && IN_AXIS_TREADY;
assign rd_en = OUT_AXIS_TVALID && OUT_AXIS_TREADY;

// reset logic
always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
        wr_ptr   <= '0;
        rd_ptr   <= '0;
        capacity <= '0;
    end
    else begin
        // write logic
        if (wr_en) begin
            wr_ptr <= wr_ptr + 1;
        end
        // read logic
        if (rd_en) begin
            rd_ptr <= rd_ptr + 1;
        end
        // capacity logic
        if (wr_en && !rd_en) begin
            capacity <= capacity + 1;
        end
        else if (!wr_en && rd_en) begin
            capacity <= capacity - 1;
        end
    end
end

endmodule
