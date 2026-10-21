// -----------------------------------------------------------------------------
// tb_pe.sv — self-checking testbench for pe.sv
//
// Run:  iverilog -g2012 -o sim/pe.vvp rtl/pe.sv tb/tb_pe.sv && vvp sim/pe.vvp
//
// Covers: reset, weight hold when load_w=0, en=0 stall, signed corner
// cases (-128 * -128, -128 * 127), and 1000 random vectors.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
`default_nettype none

module tb_pe;
    localparam int DW = 8, AW = 32;

    logic clk = 0, rst_n = 0, load_w = 0, en = 0;
    logic signed [DW-1:0] w_in = 0, a_in = 0, w_out, a_out;
    logic signed [AW-1:0] psum_in = 0, psum_out;

    int errors = 0, checks = 0;

    pe #(.DW(DW), .AW(AW)) dut (.*);

    always #5 clk = ~clk;

    task automatic load_weight(input logic signed [DW-1:0] w);
        @(negedge clk); load_w = 1; w_in = w;
        @(negedge clk); load_w = 0;
    endtask

    // Drive one MAC and compare against the reference after the clock edge.
    task automatic mac_check(input logic signed [DW-1:0] w,
                             input logic signed [DW-1:0] a,
                             input logic signed [AW-1:0] p);
        logic signed [AW-1:0] expected;
        expected = p + AW'(w * a);
        @(negedge clk); en = 1; a_in = a; psum_in = p;
        @(negedge clk); en = 0;
        checks++;
        if (psum_out !== expected || a_out !== a) begin
            errors++;
            $display("FAIL w=%0d a=%0d p=%0d -> psum=%0d (exp %0d) a_out=%0d",
                     w, a, p, psum_out, expected, a_out);
        end
    endtask

    initial begin
        logic signed [DW-1:0] w, a;
        logic signed [AW-1:0] p, held;

        // Reset
        repeat (2) @(negedge clk);
        rst_n = 1;
        checks++;
        if (psum_out !== 0 || w_out !== 0) begin errors++; $display("FAIL reset"); end

        // Signed corner cases
        load_weight(-128); mac_check(-128, -128, 0);        // +16384
        mac_check(-128, 127, 0);                             // -16256
        load_weight(127);  mac_check(127, 127, 32'sh7FFF0000);
        load_weight(0);    mac_check(0, -77, -5);

        // Weight is stationary: w_in changes with load_w low must be ignored
        load_weight(42);
        @(negedge clk); w_in = -1;
        checks++;
        if (w_out !== 42) begin errors++; $display("FAIL weight not held"); end

        // en=0 stalls: outputs hold
        mac_check(42, 3, 10);
        held = psum_out;
        @(negedge clk); a_in = 99; psum_in = 1234;
        @(negedge clk);
        checks++;
        if (psum_out !== held) begin errors++; $display("FAIL stall"); end

        // Random
        for (int i = 0; i < 1000; i++) begin
            w = $urandom; a = $urandom; p = $urandom;
            load_weight(w);
            mac_check(w, a, p);
        end

        if (errors != 0) $fatal(1, "FAIL: %0d / %0d checks failed", errors, checks);
        $display("PASS: %0d checks", checks);
        $finish;
    end
endmodule

`default_nettype wire
