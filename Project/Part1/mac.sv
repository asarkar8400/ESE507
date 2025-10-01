module mac #(
parameter INW = 16,
parameter OUTW = 64
)(
input signed [INW-1:0] input0, input1, init_value,
output logic signed [OUTW-1:0] out,
input clk, reset, init_acc, input_valid
);

logic signed [OUTW-1:0] product;

assign product = input0 * input1; //first multiplier

    always_ff @(posedge clk) begin
        if(reset) begin
            out <= 0; //synch reset
        end else if(init_acc) begin
            out <= init_value; // initialize a value
        end else if(input_valid) begin
            out <= product + out; // accumulate
        end else begin
            out <= out; // do nothing
        end      
    end
endmodule
