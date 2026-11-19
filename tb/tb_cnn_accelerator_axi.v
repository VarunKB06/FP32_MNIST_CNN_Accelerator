`timescale 1ns / 1ps
//=====================================================================
// Self-checking verification environment for the AXI4-Lite FP32 CNN accelerator
// (cnn_accelerator_axi + cnn_accelerator_top, 4x4 systolic array).
//
//   * AXI4-Lite master BFM: AW/W in any order, random delays, BREADY/RREADY back-pressure
//   * Scoreboard: independent integer model of Y = X.W (operands are multiples of 1/16, so every
//     FP32 result is exact and compared bit-for-bit), plus golden vectors from the bit-accurate
//     software model (vectors/golden_16x16.hex) for general FP32 data
//   * Procedural protocol/consistency checkers (NOT native SVA): AXI4-Lite rules and core FSM rules
//   * Directed + constrained-random tests (seeded), hand-written functional coverage
//   Run:  vvp sim +seed=N [+vcd]      Prints "FINAL: PASS" / "FINAL: FAIL".
//=====================================================================
module tb_cnn_accelerator_axi;

    localparam [10:0] ADDR_CTRL    = 11'h000;
    localparam [10:0] ADDR_SIZE    = 11'h004;
    localparam [10:0] ADDR_STATUS  = 11'h008;
    localparam [10:0] ADDR_FEATURE = 11'h00C;
    localparam [10:0] ADDR_WEIGHT  = 11'h010;
    localparam [10:0] ADDR_PERF    = 11'h014;   // 7 counters, 4 bytes apart
    localparam [10:0] ADDR_ID      = 11'h030;
    localparam [10:0] ADDR_OUT     = 11'h400;

    //-----------------------------------------------------------------
    // Clock / reset / DUT
    //-----------------------------------------------------------------
    reg clk = 1'b0;
    always #5.3125 clk = ~clk;                 // about 94.1 MHz, the clock of the implemented design

    reg ARESETN = 1'b0;
    reg [10:0] AWADDR = 0;  reg AWVALID = 0;  wire AWREADY;
    reg [31:0] WDATA = 0;   reg WVALID = 0;   wire WREADY;
    wire [1:0] BRESP;       wire BVALID;      reg BREADY = 0;
    reg [10:0] ARADDR = 0;  reg ARVALID = 0;  wire ARREADY;
    wire [31:0] RDATA;      wire [1:0] RRESP; wire RVALID; reg RREADY = 0;

    cnn_accelerator_axi dut (
        .S_AXI_ACLK(clk), .S_AXI_ARESETN(ARESETN),
        .S_AXI_AWADDR(AWADDR), .S_AXI_AWPROT(3'b000), .S_AXI_AWVALID(AWVALID), .S_AXI_AWREADY(AWREADY),
        .S_AXI_WDATA(WDATA), .S_AXI_WSTRB(4'hF), .S_AXI_WVALID(WVALID), .S_AXI_WREADY(WREADY),
        .S_AXI_BRESP(BRESP), .S_AXI_BVALID(BVALID), .S_AXI_BREADY(BREADY),
        .S_AXI_ARADDR(ARADDR), .S_AXI_ARPROT(3'b000), .S_AXI_ARVALID(ARVALID), .S_AXI_ARREADY(ARREADY),
        .S_AXI_RDATA(RDATA), .S_AXI_RRESP(RRESP), .S_AXI_RVALID(RVALID), .S_AXI_RREADY(RREADY));

    //-----------------------------------------------------------------
    // Bookkeeping
    //-----------------------------------------------------------------
    integer seed, seed0;
    integer tb_checks = 0, tb_errors = 0, tests_run = 0, tests_pass = 0, tests_fail = 0, err_base = 0;
    integer cur_id = 0, watchdog = 0, shown = 0;
    reg [8*48-1:0] cur_name;

    function integer rnd(input integer m);
        begin rnd = {$random(seed)} % m; end
    endfunction

    task check(input cond, input [8*56-1:0] msg);
        begin
            tb_checks = tb_checks + 1;
            if (!cond) begin
                tb_errors = tb_errors + 1;
                $display("[TB FAIL] t=%0t %0s", $time, msg);
            end
        end
    endtask

    //-----------------------------------------------------------------
    // AXI protocol / core consistency checkers (procedural)
    //-----------------------------------------------------------------
    integer chk_fail = 0, chk_runs = 0, aw_hs = 0, w_hs = 0, b_hs = 0, ar_hs = 0, r_hs = 0, rst_edges = 0;
    reg prev_bvalid = 0, prev_bready = 0, prev_rvalid = 0, prev_rready = 0;
    reg [31:0] prev_rdata = 0;

    task pbad(input [8*56-1:0] m);
        begin chk_fail = chk_fail + 1; $display("[CHECK FAIL] (procedural) t=%0t %0s", $time, m); end
    endtask

    always @(posedge clk) begin
        if (ARESETN) begin
            rst_edges = 0;
            chk_runs = chk_runs + 1;
            if (AWVALID && AWREADY) aw_hs = aw_hs + 1;
            if (WVALID  && WREADY)  w_hs  = w_hs  + 1;
            if (BVALID  && BREADY)  b_hs  = b_hs  + 1;
            if (ARVALID && ARREADY) ar_hs = ar_hs + 1;
            if (RVALID  && RREADY)  r_hs  = r_hs  + 1;
            if (BVALID && BRESP !== 2'b00)                   pbad("A1 BRESP is not OKAY");
            if (RVALID && RRESP !== 2'b00)                   pbad("A1 RRESP is not OKAY");
            if (prev_bvalid && !prev_bready && !BVALID)      pbad("A2 BVALID dropped before BREADY");
            if (prev_rvalid && !prev_rready &&
                (!RVALID || RDATA !== prev_rdata))           pbad("A3 RVALID/RDATA not held until RREADY");
            if (b_hs > aw_hs || b_hs > w_hs)                 pbad("A4 write response without AW and W");
            if (r_hs > ar_hs)                                pbad("A5 read data without AR handshake");
            if (BVALID && (AWREADY || WREADY))               pbad("A6 AW/W ready while response pending");
            if (^{AWREADY, WREADY, ARREADY, BVALID, RVALID} === 1'bx) pbad("A8 X/Z on AXI control outputs");
            if ((dut.load_input + dut.load_weight + dut.compute_en + dut.dataflow_load) > 1)
                                                             pbad("C1 more than one FSM phase active");
            if (dut.output_store && !dut.compute_en)         pbad("C3 output_store outside compute phase");
            prev_bvalid = BVALID; prev_bready = BREADY; prev_rvalid = RVALID; prev_rready = RREADY; prev_rdata = RDATA;
        end else begin
            rst_edges = rst_edges + 1;
            aw_hs = 0; w_hs = 0; b_hs = 0; ar_hs = 0; r_hs = 0;
            prev_bvalid = 0; prev_rvalid = 0; prev_bready = 0; prev_rready = 0;
            if (rst_edges > 1 && (BVALID !== 1'b0 || RVALID !== 1'b0)) pbad("A7 responses valid during reset");
        end
    end

    //-----------------------------------------------------------------
    // AXI4-Lite master BFM
    //-----------------------------------------------------------------
    integer t_mode = 0;     // write order: 0 AW+W together, 1 AW first, 2 W first
    integer t_gap  = 0;     // cycles between the two phases when ordered
    integer t_bd   = 0;     // BREADY delay (back-pressure on B)
    integer t_rd   = 0;     // RREADY delay (back-pressure on R)

    task aw_send(input [10:0] a);
        begin
            @(posedge clk); AWADDR <= a; AWVALID <= 1'b1;
            @(posedge clk); while (!AWREADY) @(posedge clk);
            AWVALID <= 1'b0;
        end
    endtask

    task w_send(input [31:0] d);
        begin
            @(posedge clk); WDATA <= d; WVALID <= 1'b1;
            @(posedge clk); while (!WREADY) @(posedge clk);
            WVALID <= 1'b0;
        end
    endtask

    task b_phase;
        begin
            @(posedge clk); while (!BVALID) @(posedge clk);
            repeat (t_bd) @(posedge clk);
            BREADY <= 1'b1;
            @(posedge clk);
            BREADY <= 1'b0;
        end
    endtask

    task axi_write(input [10:0] a, input [31:0] d);
        begin
            if (t_mode == 1)      begin aw_send(a); repeat (t_gap) @(posedge clk); w_send(d); end
            else if (t_mode == 2) begin w_send(d); repeat (t_gap) @(posedge clk); aw_send(a); end
            else                  begin fork aw_send(a); w_send(d); join end
            b_phase;
        end
    endtask

    task axi_read(input [10:0] a, output [31:0] d);
        begin
            @(posedge clk); ARADDR <= a; ARVALID <= 1'b1;
            @(posedge clk); while (!ARREADY) @(posedge clk);
            ARVALID <= 1'b0;
            @(posedge clk); while (!RVALID) @(posedge clk);
            repeat (t_rd) @(posedge clk);
            RREADY <= 1'b1;
            d = RDATA;
            @(posedge clk);
            RREADY <= 1'b0;
        end
    endtask

    task timing_default; begin t_mode = 0; t_gap = 0; t_bd = 0; t_rd = 0; end endtask
    task timing_random;
        begin t_mode = rnd(3); t_gap = rnd(4); t_bd = rnd(4); t_rd = rnd(4); end
    endtask

    task hw_reset;
        begin
            ARESETN <= 1'b0; repeat (5) @(posedge clk);
            ARESETN <= 1'b1; repeat (3) @(posedge clk);
        end
    endtask

    //-----------------------------------------------------------------
    // FP32 helpers: value = v / 2**shift, exact for |v| < 2**24
    //-----------------------------------------------------------------
    function [31:0] fx_to_fp32(input integer v, input integer shift);
        integer mag, p, i;
        reg [31:0] mant;
        reg [7:0]  e;
        reg        sgn;
        begin
            if (v == 0) fx_to_fp32 = 32'h0000_0000;
            else begin
                sgn = (v < 0);
                mag = sgn ? -v : v;
                p = 0;
                for (i = 0; i < 31; i = i + 1) if ((mag >> i) != 0) p = i;
                e    = p + 127 - shift;
                mant = (mag << (23 - p)) & 32'h007F_FFFF;
                fx_to_fp32 = {sgn, e, mant[22:0]};
            end
        end
    endfunction

    function same_value(input [31:0] a, input [31:0] b);     // bit-equal, or both +-0
        begin same_value = (a === b) || ((a[30:0] === 31'd0) && (b[30:0] === 31'd0)); end
    endfunction

    //-----------------------------------------------------------------
    // Data, scoreboard
    //-----------------------------------------------------------------
    integer xi [0:255];  integer wi [0:255];  integer yexp [0:255];
    reg [31:0] xw [0:255]; reg [31:0] ww [0:255]; reg [31:0] yact [0:255]; reg [31:0] yprev [0:255];
    reg [31:0] gv [0:6143];                       // 8 sets x (X, W, Y) x 256 words
    integer last_L = 4, last_cls = 0;

    function integer size_code(input integer L);
        begin size_code = (L == 4) ? 0 : ((L == 8) ? 1 : 2); end
    endfunction

    // data classes (all operands are multiples of 1/16 in [-8, 8))
    // 0 random  1 zeros  2 identity W  3 negative  4 sparse  5 max magnitude  6 cancellation
    task gen_data(input integer L, input integer cls);
        integer m, k, n;
        begin
            for (m = 0; m < L*L; m = m + 1) begin
                xi[m] = rnd(256) - 128; wi[m] = rnd(256) - 128;
            end
            case (cls)
                1: for (m = 0; m < L*L; m = m + 1) begin xi[m] = 0; wi[m] = 0; end
                2: for (k = 0; k < L; k = k + 1) for (n = 0; n < L; n = n + 1) wi[k*L+n] = (k == n) ? 16 : 0;
                3: for (m = 0; m < L*L; m = m + 1) begin xi[m] = -1 - rnd(128); wi[m] = -1 - rnd(128); end
                4: for (m = 0; m < L*L; m = m + 1) begin
                       if (rnd(10) != 0) xi[m] = 0;
                       if (rnd(10) != 0) wi[m] = 0;
                   end
                5: for (m = 0; m < L*L; m = m + 1) begin xi[m] = -128; wi[m] = -128; end
                6: for (k = 0; k < L; k = k + 1) for (n = 0; n < L; n = n + 1) begin
                       wi[k*L+n] = (k % 2 == 0) ? 37 : -37;       // every row of X is constant: sums cancel
                       if (n == 0) for (m = 0; m < L; m = m + 1) xi[m*L+k] = 53;
                   end
                default: ;
            endcase
        end
    endtask

    task ints_to_words(input integer L);
        integer i;
        begin
            for (i = 0; i < L*L; i = i + 1) begin xw[i] = fx_to_fp32(xi[i], 4); ww[i] = fx_to_fp32(wi[i], 4); end
        end
    endtask

    task compute_expected(input integer L);       // independent integer model: Y = X . W
        integer m, n, k, s;
        begin
            for (m = 0; m < L; m = m + 1)
                for (n = 0; n < L; n = n + 1) begin
                    s = 0;
                    for (k = 0; k < L; k = k + 1) s = s + xi[m*L+k] * wi[k*L+n];
                    yexp[m*L+n] = s;
                end
        end
    endtask

    task check_Y_exact(input integer L);
        integer i, bad;
        reg [31:0] e;
        begin
            bad = 0;
            for (i = 0; i < L*L; i = i + 1) begin
                e = fx_to_fp32(yexp[i], 8);
                tb_checks = tb_checks + 1;
                if (!same_value(yact[i], e)) begin
                    bad = bad + 1; tb_errors = tb_errors + 1;
                    if (bad <= 4) $display("[TB FAIL] t=%0t Y[%0d][%0d] got 0x%08h expected 0x%08h (%0dx%0d)",
                                            $time, i / L, i % L, yact[i], e, L, L);
                end
            end
        end
    endtask

    //-----------------------------------------------------------------
    // Accelerator operations through the AXI interface
    //-----------------------------------------------------------------
    reg [31:0] rv;
    reg [31:0] perf [0:6];

    task load_X(input integer L); integer i; begin for (i = 0; i < L*L; i = i + 1) axi_write(ADDR_FEATURE, xw[i]); end endtask
    task load_W(input integer L); integer i; begin for (i = 0; i < L*L; i = i + 1) axi_write(ADDR_WEIGHT,  ww[i]); end endtask
    task start_run;   begin axi_write(ADDR_CTRL, 32'h1); repeat (3) @(posedge clk); end endtask
    task soft_reset;  begin axi_write(ADDR_CTRL, 32'h2); repeat (3) @(posedge clk); end endtask

    task wait_done;
        integer n;
        begin
            n = 0; rv = 0;
            while (!rv[0] && n < 4000) begin axi_read(ADDR_STATUS, rv); n = n + 1; end
            check(rv[0] === 1'b1, "run finished (STATUS.done) before timeout");
        end
    endtask

    task read_Y(input integer L);
        integer i;
        begin for (i = 0; i < L*L; i = i + 1) axi_read(ADDR_OUT + 4*i, yact[i]); end
    endtask

    task read_perf; integer i; begin for (i = 0; i < 7; i = i + 1) axi_read(ADDR_PERF + 4*i, perf[i]); end endtask

    //-----------------------------------------------------------------
    // Functional coverage (hand-written bins; NOT code coverage)
    //-----------------------------------------------------------------
    reg [2:0] cv_size = 0;                 // 4, 8, 16
    reg [6:0] cv_cls  = 0;                 // exact data classes
    reg [2:0] cv_mode = 0;                 // AW+W, AW first, W first
    reg cv_bp_b = 0, cv_bp_r = 0, cv_soft = 0, cv_hard = 0, cv_reuse = 0, cv_rerun = 0, cv_startfirst = 0,
        cv_busywr = 0, cv_b2b = 0, cv_ro = 0, cv_perf = 0, cv_golden = 0, cv_mixed = 0;
    reg [20:0] cx_size_cls = 0;            // size x class
    reg [8:0]  cx_mode_size = 0;           // write order x size
    integer sidx;

    task cov_run(input integer L, input integer cls);
        begin
            sidx = (L == 4) ? 0 : ((L == 8) ? 1 : 2);
            cv_size[sidx] = 1'b1;
            if (cls >= 0 && cls < 7) begin cv_cls[cls] = 1'b1; cx_size_cls[sidx*7 + cls] = 1'b1; end
            if (t_mode >= 0 && t_mode < 3) begin cv_mode[t_mode] = 1'b1; cx_mode_size[t_mode*3 + sidx] = 1'b1; end
            if (t_bd > 0) cv_bp_b = 1'b1;
            if (t_rd > 0) cv_bp_r = 1'b1;
        end
    endtask

    function integer count_bits(input [31:0] v, input integer n);
        integer i, c;
        begin c = 0; for (i = 0; i < n; i = i + 1) if (v[i]) c = c + 1; count_bits = c; end
    endfunction

    function integer evt_hit(input integer dummy);
        begin
            evt_hit = count_bits(cv_size, 3) + count_bits(cv_cls, 7) + count_bits(cv_mode, 3) + cv_bp_b + cv_bp_r + cv_soft +
                      cv_hard + cv_reuse + cv_rerun + cv_startfirst + cv_busywr + cv_b2b + cv_ro + cv_perf + cv_golden + cv_mixed;
        end
    endfunction
    function integer evt_total(input integer dummy); begin evt_total = 3 + 7 + 3 + 13; end endfunction
    function integer x_hit(input integer dummy); begin x_hit = count_bits(cx_size_cls, 21) + count_bits(cx_mode_size, 9); end endfunction
    function integer x_total(input integer dummy); begin x_total = 21 + 9; end endfunction

    //-----------------------------------------------------------------
    // Test bookkeeping
    //-----------------------------------------------------------------
    function integer errors_now(input integer dummy); begin errors_now = tb_errors + chk_fail; end endfunction

    task begin_test(input integer id, input [8*48-1:0] name);
        begin
            cur_id = id; cur_name = name; err_base = errors_now(0);
            $display("[T%0d] START %0s", id, name);
            timing_default; hw_reset;
        end
    endtask

    task end_test;
        integer d;
        begin
            d = errors_now(0) - err_base;
            tests_run = tests_run + 1;
            if (d == 0) begin tests_pass = tests_pass + 1; $display("[T%0d] PASS  %0s", cur_id, cur_name); end
            else begin tests_fail = tests_fail + 1; $display("[T%0d] FAIL  %0s (%0d new errors)", cur_id, cur_name, d); end
        end
    endtask

    // one complete exact-data run
    task run_exact(input integer L, input integer cls, input reuse_w);
        begin
            if (reuse_w) begin : reuse_blk
                integer i;
                for (i = 0; i < L*L; i = i + 1) xi[i] = rnd(256) - 128;     // new X, W kept
                ints_to_words(L);
            end else begin
                gen_data(L, cls); ints_to_words(L);
            end
            compute_expected(L);
            axi_write(ADDR_SIZE, size_code(L));
            load_X(L);
            if (!reuse_w) load_W(L);
            start_run; wait_done; read_Y(L);
            check_Y_exact(L);
            cov_run(L, cls); last_L = L; last_cls = cls;
        end
    endtask

    //-----------------------------------------------------------------
    // Tests
    //-----------------------------------------------------------------
    integer i, j, L, cls, n, pass_cnt;
    reg [31:0] r0, r1;
    integer exp_perf [0:20];

    task test_reset_values;
        begin
            begin_test(1, "reset values and hardware ID");
            axi_read(ADDR_ID, rv);     check(rv === 32'h4D4D3031, "ID register reads 0x4D4D3031");
            axi_read(ADDR_SIZE, rv);   check(rv === 32'd0, "SIZE resets to 0");
            axi_read(ADDR_STATUS, rv); check(rv[3:0] === 4'b0000, "STATUS done/busy/loaded flags clear");
            read_perf;
            for (i = 0; i < 7; i = i + 1) check(perf[i] === 32'd0, "performance counters reset to 0");
            axi_read(ADDR_OUT, rv);          check(rv === 32'd0, "output window starts at 0 (first word)");
            axi_read(ADDR_OUT + 11'h3FC, rv); check(rv === 32'd0, "output window starts at 0 (last word)");
            end_test;
        end
    endtask

    task test_register_access;
        begin
            begin_test(2, "register read/write, RO and unmapped writes");
            for (i = 0; i < 4; i = i + 1) begin
                axi_write(ADDR_SIZE, i); axi_read(ADDR_SIZE, rv);
                check(rv === i, "SIZE write/readback");
            end
            axi_write(ADDR_SIZE, 32'd1);
            axi_write(ADDR_ID, 32'hDEAD_BEEF);      axi_read(ADDR_ID, rv);     check(rv === 32'h4D4D3031, "ID is read-only");
            axi_write(ADDR_STATUS, 32'hFFFF_FFFF);  axi_read(ADDR_STATUS, rv); check(rv[3:0] === 4'b0000, "STATUS is read-only");
            axi_write(ADDR_PERF, 32'h1234_5678);    axi_read(ADDR_PERF, rv);   check(rv === 32'd0, "perf counters are read-only");
            axi_write(11'h100, 32'hCAFE_F00D);      axi_read(11'h100, rv);     check(rv === 32'd0, "unmapped address reads 0");
            axi_read(ADDR_SIZE, rv);                check(rv === 32'd1, "unmapped/RO writes did not disturb SIZE");
            cv_ro = 1'b1;
            end_test;
        end
    endtask

    task test_identity_4x4;
        begin
            begin_test(3, "4x4 identity (original basic test)");
            for (i = 0; i < 16; i = i + 1) begin xi[i] = (i + 1) * 3 - 20; wi[i] = (i / 4 == i % 4) ? 16 : 0; end
            ints_to_words(4); compute_expected(4);
            axi_write(ADDR_SIZE, 0); load_X(4); load_W(4); start_run; wait_done; read_Y(4);
            check_Y_exact(4);
            for (i = 0; i < 16; i = i + 1) check(same_value(yact[i], fx_to_fp32(xi[i], 4)), "Y equals X for identity W");
            cov_run(4, 2);
            end_test;
        end
    endtask

    task test_directed_4x4;
        begin
            begin_test(4, "4x4 X rows [1 2 3 4], W all ones");
            for (i = 0; i < 16; i = i + 1) begin xi[i] = ((i % 4) + 1) * 16; wi[i] = 16; end
            ints_to_words(4); compute_expected(4);
            axi_write(ADDR_SIZE, 0); load_X(4); load_W(4); start_run; wait_done; read_Y(4);
            check_Y_exact(4);
            for (i = 0; i < 16; i = i + 1) check(yact[i] === 32'h4120_0000, "every output is 10.0");
            end_test;
        end
    endtask

    task test_size_class_sweep;
        begin
            begin_test(5, "3 sizes x 7 data classes (exact, bit-for-bit)");
            for (L = 4; L <= 16; L = L * 2)
                for (cls = 0; cls < 7; cls = cls + 1) run_exact(L, cls, 1'b0);
            end_test;
        end
    endtask

    task test_perf_counters;
        begin
            begin_test(6, "performance counters match board measurements");
            // order: total, feature load, weight load, compute, output store, busy, idle (measured on the PYNQ-Z2)
            exp_perf[0]=64;   exp_perf[1]=1; exp_perf[2]=1; exp_perf[3]=48;   exp_perf[4]=16;   exp_perf[5]=54;   exp_perf[6]=10;
            exp_perf[7]=287;  exp_perf[8]=1; exp_perf[9]=1; exp_perf[10]=256; exp_perf[11]=128; exp_perf[12]=274; exp_perf[13]=13;
            exp_perf[14]=1627;exp_perf[15]=1;exp_perf[16]=1;exp_perf[17]=1536;exp_perf[18]=1024;exp_perf[19]=1602;exp_perf[20]=25;
            for (j = 0; j < 3; j = j + 1) begin
                L = 4 << j;
                run_exact(L, 0, 1'b0);
                read_perf;
                for (i = 0; i < 7; i = i + 1) begin
                    check(perf[i] === exp_perf[j*7+i], "perf counter equals the measured value");
                    if (perf[i] !== exp_perf[j*7+i])
                        $display("   size %0d counter %0d: got %0d expected %0d", L, i, perf[i], exp_perf[j*7+i]);
                end
                check(perf[0] === perf[5] + perf[6], "total cycles = busy + idle");
            end
            cv_perf = 1'b1;
            end_test;
        end
    endtask

    task test_back_to_back;
        begin
            begin_test(7, "back-to-back runs, mixed sizes");
            for (i = 0; i < 6; i = i + 1) run_exact(4, 0, 1'b0);
            run_exact(8, 0, 1'b0); run_exact(16, 0, 1'b0); run_exact(4, 4, 1'b0); run_exact(16, 3, 1'b0); run_exact(8, 6, 1'b0);
            cv_b2b = 1'b1; cv_mixed = 1'b1;
            end_test;
        end
    endtask

    task test_weight_reuse;
        begin
            begin_test(8, "weights loaded once, three different X");
            run_exact(16, 0, 1'b0);
            for (i = 0; i < 3; i = i + 1) run_exact(16, 0, 1'b1);
            run_exact(8, 0, 1'b0);
            for (i = 0; i < 3; i = i + 1) run_exact(8, 0, 1'b1);
            cv_reuse = 1'b1;
            end_test;
        end
    endtask

    task test_rerun_no_reload;
        begin
            begin_test(9, "START again without rewriting X or W");
            run_exact(8, 0, 1'b0);
            for (i = 0; i < 64; i = i + 1) yprev[i] = yact[i];
            read_perf; r0 = perf[0];
            start_run; wait_done; read_Y(8);
            for (i = 0; i < 64; i = i + 1) check(yact[i] === yprev[i], "second run gives the same result");
            read_perf; check(perf[0] === r0, "second run takes the same number of cycles");
            cv_rerun = 1'b1;
            end_test;
        end
    endtask

    task test_start_before_load;
        begin
            begin_test(10, "START issued before the data is loaded");
            L = 8; gen_data(L, 0); ints_to_words(L); compute_expected(L);
            axi_write(ADDR_SIZE, size_code(L));
            soft_reset;
            start_run;                                   // FSM waits in LOAD_INPUT
            axi_read(ADDR_STATUS, rv); check(rv[1] === 1'b0 || rv[1] === 1'b1, "STATUS readable while waiting");
            load_X(L); load_W(L);
            wait_done; read_Y(L);
            check_Y_exact(L);
            cov_run(L, 0); cv_startfirst = 1'b1;
            end_test;
        end
    endtask

    task test_writes_while_busy;
        begin
            begin_test(11, "data writes are ignored while a run is active");
            L = 16; gen_data(L, 0); ints_to_words(L); compute_expected(L);
            axi_write(ADDR_SIZE, size_code(L)); load_X(L); load_W(L);
            start_run;
            rv = 0; n = 0;
            while (!rv[1] && n < 50) begin axi_read(ADDR_STATUS, rv); n = n + 1; end
            check(rv[1] === 1'b1, "STATUS.busy observed during the run");
            for (i = 0; i < 24; i = i + 1) begin axi_write(ADDR_FEATURE, 32'h7F7F_7F7F); axi_write(ADDR_WEIGHT, 32'h7F7F_7F7F); end
            wait_done; read_Y(L);
            check_Y_exact(L);
            cov_run(L, 0); cv_busywr = 1'b1;
            end_test;
        end
    endtask

    task test_soft_reset_in_run;
        begin
            begin_test(12, "soft reset (CTRL bit 1) during a run, then recovery");
            L = 16; gen_data(L, 0); ints_to_words(L); compute_expected(L);
            axi_write(ADDR_SIZE, size_code(L)); load_X(L); load_W(L);
            start_run; repeat (200) @(posedge clk);
            soft_reset;
            axi_read(ADDR_STATUS, rv); check(rv[3:0] === 4'b0000, "STATUS clear after soft reset");
            read_perf; for (i = 0; i < 7; i = i + 1) check(perf[i] === 32'd0, "perf counters cleared by soft reset");
            run_exact(L, 0, 1'b0);
            cv_soft = 1'b1;
            end_test;
        end
    endtask

    task test_hard_reset_in_run;
        begin
            begin_test(13, "ARESETN low during a run, then recovery");
            L = 8; gen_data(L, 0); ints_to_words(L); compute_expected(L);
            axi_write(ADDR_SIZE, size_code(L)); load_X(L); load_W(L);
            start_run; repeat (60) @(posedge clk);
            hw_reset;
            axi_read(ADDR_SIZE, rv);   check(rv === 32'd0, "SIZE cleared by hardware reset");
            axi_read(ADDR_STATUS, rv); check(rv[3:0] === 4'b0000, "STATUS clear after hardware reset");
            axi_read(ADDR_ID, rv);     check(rv === 32'h4D4D3031, "ID readable after reset");
            run_exact(8, 0, 1'b0);
            cv_hard = 1'b1;
            end_test;
        end
    endtask

    task test_axi_write_orders;
        begin
            begin_test(14, "AW/W arrival order and delays (SIZE register)");
            for (i = 0; i < 60; i = i + 1) begin
                timing_random;
                j = rnd(4);
                axi_write(ADDR_SIZE, j); axi_read(ADDR_SIZE, rv);
                check(rv === j, "address and data paired correctly");
                cv_mode[t_mode] = 1'b1;
                if (t_bd > 0) cv_bp_b = 1'b1;
                if (t_rd > 0) cv_bp_r = 1'b1;
            end
            end_test;
        end
    endtask

    task test_backpressure;
        begin
            begin_test(15, "B/R back-pressure while loading and reading 16x16");
            t_mode = 1; t_gap = 2; t_bd = 3; t_rd = 2;
            run_exact(16, 0, 1'b0);
            t_mode = 2; t_gap = 1; t_bd = 1; t_rd = 4;
            run_exact(8, 0, 1'b0);
            end_test;
        end
    endtask

    task test_golden;
        reg [31:0] e;
        begin
            begin_test(16, "golden vectors (software-model bit-exact, FP32)");
            check(^gv[0] !== 1'bx, "vectors/golden_16x16.hex loaded");
            for (j = 0; j < 8; j = j + 1) begin
                for (i = 0; i < 256; i = i + 1) begin xw[i] = gv[j*768 + i]; ww[i] = gv[j*768 + 256 + i]; end
                axi_write(ADDR_SIZE, 2); load_X(16); load_W(16); start_run; wait_done; read_Y(16);
                pass_cnt = 0;
                for (i = 0; i < 256; i = i + 1) begin
                    e = gv[j*768 + 512 + i];
                    tb_checks = tb_checks + 1;
                    if (!same_value(yact[i], e)) begin
                        tb_errors = tb_errors + 1;
                        if (pass_cnt < 4) $display("[TB FAIL] t=%0t golden set %0d Y[%0d][%0d] got 0x%08h expected 0x%08h",
                                                    $time, j, i / 16, i % 16, yact[i], e);
                        pass_cnt = pass_cnt + 1;
                    end
                end
                cov_run(16, -1);
            end
            cv_golden = 1'b1;
            end_test;
        end
    endtask

    task test_random_stress;
        begin
            begin_test(17, "constrained-random stress (seeded)");
            for (n = 0; n < 40; n = n + 1) begin
                timing_random;
                L = 4 << rnd(3);
                cls = rnd(7);
                run_exact(L, cls, (n > 0) && (rnd(4) == 0) && (L == last_L));
            end
            end_test;
        end
    endtask

    //-----------------------------------------------------------------
    // Summary / main
    //-----------------------------------------------------------------
    task summary;
        reg ok;
        begin
            $display("=== FUNCTIONAL COVERAGE (hand-written bins; not code coverage) ===");
            $display("sizes 4/8/16 hit: %0b%0b%0b", cv_size[0], cv_size[1], cv_size[2]);
            $display("data classes random/zeros/identity/negative/sparse/maxmag/cancel hit: %0b%0b%0b%0b%0b%0b%0b",
                     cv_cls[0], cv_cls[1], cv_cls[2], cv_cls[3], cv_cls[4], cv_cls[5], cv_cls[6]);
            $display("AXI write order AW+W / AW first / W first: %0b%0b%0b", cv_mode[0], cv_mode[1], cv_mode[2]);
            $display("B back-pressure %0b, R back-pressure %0b, soft reset in run %0b, hardware reset in run %0b",
                     cv_bp_b, cv_bp_r, cv_soft, cv_hard);
            $display("weight reuse %0b, rerun without reload %0b, start before load %0b, writes while busy %0b",
                     cv_reuse, cv_rerun, cv_startfirst, cv_busywr);
            $display("back-to-back %0b, mixed sizes %0b, RO/unmapped writes %0b, perf counters %0b, golden vectors %0b",
                     cv_b2b, cv_mixed, cv_ro, cv_perf, cv_golden);
            $display("EVENT BINS COVERED: %0d / %0d", evt_hit(0), evt_total(0));
            $display("CROSS BINS (size x class 21, write order x size 9) COVERED: %0d / %0d", x_hit(0), x_total(0));
            $display("=== END ===");
            $display("=== PROTOCOL CHECKER SUMMARY ===");
            $display("mode: PROCEDURAL checkers (not native SVA)");
            $display("checker failures: %0d  (evaluations: %0d)", chk_fail, chk_runs);
            $display("AXI handshakes: AW %0d, W %0d, B %0d, AR %0d, R %0d", aw_hs, w_hs, b_hs, ar_hs, r_hs);
            $display("=== END ===");
            ok = (tests_run > 0) && (tests_fail == 0) && (tb_errors == 0) && (chk_fail == 0) && !watchdog;
            $display("=== SUMMARY ===");
            $display("seed: %0d   test: 0", seed0);
            $display("total tests: %0d", tests_run);
            $display("passed tests: %0d", tests_pass);
            $display("failed tests: %0d", tests_fail);
            $display("total checks: %0d", tb_checks);
            $display("scoreboard/testbench errors: %0d", tb_errors);
            $display("protocol checker failures: %0d", chk_fail);
            $display("functional EVENT bins covered: %0d / %0d", evt_hit(0), evt_total(0));
            $display("functional CROSS bins covered: %0d / %0d", x_hit(0), x_total(0));
            if (watchdog) $display("WATCHDOG TIMEOUT");
            $display("FINAL: %0s", ok ? "PASS" : "FAIL");
            $display("=== END ===");
        end
    endtask

    initial begin
        repeat (2000) #1_000_000;                       // 2 s of simulated time
        watchdog = 1; $display("[WATCHDOG] simulation timeout"); summary; $finish;
    end

    initial begin
        if (!$value$plusargs("seed=%d", seed)) seed = 1;
        seed0 = seed;
        if ($test$plusargs("vcd")) begin
            $dumpfile("waveforms/tb_cnn_accelerator_axi.vcd");
            $dumpvars(0, tb_cnn_accelerator_axi);
        end
        $readmemh("vectors/golden_16x16.hex", gv);
        $display("=== tb_cnn_accelerator_axi: seed=%0d ===", seed0);
        repeat (5) @(posedge clk);
        test_reset_values;
        test_register_access;
        test_identity_4x4;
        test_directed_4x4;
        test_size_class_sweep;
        test_perf_counters;
        test_back_to_back;
        test_weight_reuse;
        test_rerun_no_reload;
        test_start_before_load;
        test_writes_while_busy;
        test_soft_reset_in_run;
        test_hard_reset_in_run;
        test_axi_write_orders;
        test_backpressure;
        test_golden;
        test_random_stress;
        summary;
        $finish;
    end
endmodule
