// Aritro Sarkar
// ESE 507HW4 (Fall 2025)

// START OF DESIGN
module register_8bit #(parameter DATA_WIDTH = 8)
(
    input unsigned [DATA_WIDTH-1:0] din,
    input load,
    input add,
    input clk,
    input reset_n,
    output logic unsigned [DATA_WIDTH-1:0] dout
);

logic unsigned [DATA_WIDTH-1:0] dtemp;

    always_ff @(posedge clk) begin
        if(~reset_n) begin
            dtemp = 0;
        end else if(load == 1) begin
            dtemp = din;
        end else if(add == 1) begin
            dtemp = dtemp + 1;
        end
    end

assign dout = dtemp;

endmodule

// END OF DESIGN

//---------------------------------------------------------------------------------------------------
// START OF TB
module test();

   // Testbench Input Signals
   logic unsigned [7:0] din_tb;
   logic                load_tb, add_tb, clk_tb, reset_n_tb;

   // Testbench Output Signal
   logic unsigned [7:0] dout_tb;

    // Clock generation
   initial clk_tb=0;
   always #5 clk_tb = ~clk_tb;

   // Instantiate the DUT
   register_8bit register_8bit_inst(.din(din_tb), .load(load_tb), .add(add_tb), .clk(clk_tb), .reset_n(reset_n_tb), .dout(dout_tb));

   // Now, use an initial block to tell testbench what to do.
   initial begin
      $monitor($time,"din = %d , clk = %d, reset_n = %d, load = %d, add = %d, dout = %d", din_tb, clk_tb, reset_n_tb, load_tb, add_tb, dout_tb);

      @(posedge clk_tb); // 5ns
      #1; // 6ns
      din_tb     = 0;
      reset_n_tb = 0; // reset active but register iwll actually reset on the next positive clk edge
      load_tb    = 0;
      add_tb     = 0;

      @(posedge clk_tb);
      #1; // 16ns
      reset_n_tb = 1;

      // Load a value
      @(posedge clk_tb); #1;
      din_tb  = 8'd72;
      load_tb = 1;             

      // Increment 
      @(posedge clk_tb); #1; 
      load_tb = 0;             
      add_tb = 1;
      
      @(posedge clk_tb); #21; add_tb = 0;
     
      // Load another value 
      @(posedge clk_tb); #1;
      din_tb  = 8'd100;
      load_tb = 1;
      add_tb  = 1;
      @(posedge clk_tb); #1;
      load_tb = 0;
      @(posedge clk_tb); #16; //will add +2 to 102
      add_tb  = 0;
      

      // Overflow check
      @(posedge clk_tb); #1;
      din_tb  = 8'd254;
      load_tb = 1;
      
      @(posedge clk_tb); #1;
      load_tb = 0;
      add_tb  = 1;
      
      @(posedge clk_tb); #21;  // this will add +3 to 254 and end up wrapping to d1
      add_tb = 0;

      $finish;
   end
endmodule
//END OF TB
