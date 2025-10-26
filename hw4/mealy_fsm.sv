module mealy_fsm(input clk, reset, X, output logic Y);

    // State assignment using enums:
    enum logic [1:0] {A, B, C} state, next_state;

     // state transitions and output logic
    always_comb begin
        next_state = state; // the default
        Y = 0;
        case (state)
            A: begin
                if (X == 0) begin
                    next_state = A;   // A -> A
                    Y = 1;
                end else begin
                    next_state = B;   // A -> B
                    Y = 0;
                end
            end

            B: begin
                if (X == 0) begin
                    next_state = A;   // B -> A
                    Y = 1;
                end else begin
                    next_state = C;   // B -> C
                    Y = 0;
                end
            end

            C: begin
                if (X == 0) begin
                    next_state = B;   // C -> B
                    Y = 0;
                end else begin
                    next_state = A;   // C -> A
                    Y = 1;
                end
            end

            default: begin // just in case state os some number that isnt 0-2
                next_state = B;
                Y = 0;
            end
        endcase
    end

   // reset logic
    always_ff @(posedge clk) begin
        if (reset)
            state <= B; 
        else
            state <= next_state;
        end

endmodule

module test();
    logic clk, reset, X, Y;

    initial clk=0;
    always #5 clk = ~clk;

    mealy_fsm dut(.clk(clk), .reset(reset), .X(X), .Y(Y));

    initial begin
        X = 0;
        // reset to B
        reset = 1;
        @(posedge clk); #1;
        reset = 0;

        // B and X=1 -> C and Y=0
        @(posedge clk); #1; X = 1;
        
        // C and X=1 -> A and Y=1
        @(posedge clk); #1; X = 1;
        
        // A and X=1 -> B and Y=0
        @(posedge clk); #1; X = 1;

        // B and X=0 -> A and Y=1
        @(posedge clk); #1; X = 0;

        // A and X=0 -> A and Y=1
        @(posedge clk); #1; X = 0;

        // A and X=1 -> B and Y=0
        @(posedge clk); #1; X = 1;

        // B and X=0 -> A and Y=1
        @(posedge clk); #1; X = 0;

        // A and X=1 -> B and Y=0
        @(posedge clk); #1; X = 1;

        // B with X=1 -> C and Y=0
        @(posedge clk); #1; X = 1;

        // C and X=0 -> B and Y=0
        @(posedge clk); #1; X = 0;

        // B and X=1 -> C and Y=0
        @(posedge clk); #1; X = 1;
        $stop;
    end
endmodule
