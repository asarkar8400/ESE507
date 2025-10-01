module mac_pipe #(
parameter INW = 16,
parameter OUTW = 64
)(
input signed [INW-1:0] input0, input1, init_value,
output logic signed [OUTW-1:0] out,
input clk, reset, init_acc, input_valid
);

logic signed [OUTW-1:0] product, q;
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
