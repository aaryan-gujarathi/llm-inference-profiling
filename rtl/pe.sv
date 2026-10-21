// -----------------------------------------------------------------------------
// pe.sv — weight-stationary systolic processing element
//
// One PE of the 8x8 array. The weight is loaded once and held; activations
// flow left->right, partial sums flow top->bottom.
//
//   a_in  ──►[ w ]──► a_out          psum_out = psum_in + w * a_in
//              │
//   psum_in ▼  ▼ psum_out
//
// Datapath: INT8 x INT8 -> INT16 product, INT32 accumulate. With K = 2048
// the worst-case column sum is 2048 * 128 * 128 = 2^25, well inside INT32,
// so no saturation is needed here (requantize at the array edge).
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
`default_nettype none

module pe #(
    parameter int DW = 8,   // activation / weight width (signed)
    parameter int AW = 32   // accumulator width (signed)
) (
    input  wire                 clk,
    input  wire                 rst_n,     // active-low synchronous reset
    // weight load: shifts down the column while load_w is high
    input  wire                 load_w,
    input  wire signed [DW-1:0] w_in,
    output logic signed [DW-1:0] w_out,
    // compute
    input  wire                 en,        // advance the pipeline
    input  wire signed [DW-1:0] a_in,
    input  wire signed [AW-1:0] psum_in,
    output logic signed [DW-1:0] a_out,
    output logic signed [AW-1:0] psum_out
);

    logic signed [DW-1:0]   w_reg;
    logic signed [2*DW-1:0] prod;

    assign w_out = w_reg;
    assign prod  = w_reg * a_in;   // both operands signed -> signed multiply

    // Weight register: loads only when load_w is asserted (stationary otherwise).
    always_ff @(posedge clk) begin
        if (!rst_n)      w_reg <= '0;
        else if (load_w) w_reg <= w_in;
    end

    // Activation forward + MAC, one cycle latency each.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_out    <= '0;
            psum_out <= '0;
        end else if (en) begin
            a_out    <= a_in;
            psum_out <= psum_in + AW'(prod);   // sign-extends the product
        end
    end

endmodule

`default_nettype wire
