
module fp32_mul (
    input  wire        clk,
    input  wire        clr,          // synchronous clear of all stages
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] p
);

    wire [7:0] ea = a[30:23];
    wire [7:0] eb = b[30:23];

    // ---- stage 1 : unpack
    reg        s1_sign, s1_zero;
    reg [9:0]  s1_exp;
    reg [23:0] s1_ma, s1_mb;

    always @(posedge clk) begin
        if (clr) begin
            s1_sign <= 1'b0;
            s1_zero <= 1'b1;
            s1_exp  <= 10'd0;
            s1_ma   <= 24'd0;
            s1_mb   <= 24'd0;
        end
        else begin
            s1_sign <= a[31] ^ b[31];
            s1_zero <= (ea == 8'd0) || (eb == 8'd0);
            s1_exp  <= {2'b00, ea} + {2'b00, eb};
            s1_ma   <= {1'b1, a[22:0]};
            s1_mb   <= {1'b1, b[22:0]};
        end
    end

    // ---- stage 2 : 24 x 24 mantissa product (maps to DSP48E1)
    (* use_dsp = "yes" *) reg [47:0] s2_prod;
    reg        s2_sign, s2_zero;
    reg [9:0]  s2_exp;

    always @(posedge clk) begin
        if (clr) begin
            s2_prod <= 48'd0;
            s2_sign <= 1'b0;
            s2_zero <= 1'b1;
            s2_exp  <= 10'd0;
        end
        else begin
            s2_prod <= s1_ma * s1_mb;
            s2_sign <= s1_sign;
            s2_zero <= s1_zero;
            s2_exp  <= s1_exp;
        end
    end

    // ---- stage 3 : pipeline register (lets the DSP use its P register)
    reg [47:0] s3_prod;
    reg        s3_sign, s3_zero;
    reg [9:0]  s3_exp;

    always @(posedge clk) begin
        if (clr) begin
            s3_prod <= 48'd0;
            s3_sign <= 1'b0;
            s3_zero <= 1'b1;
            s3_exp  <= 10'd0;
        end
        else begin
            s3_prod <= s2_prod;
            s3_sign <= s2_sign;
            s3_zero <= s2_zero;
            s3_exp  <= s2_exp;
        end
    end

    // ---- stage 4 : normalise, round (nearest-even), pack
    reg [23:0] n_m;
    reg        n_g, n_st, n_up;
    reg [24:0] n_mr;
    reg [9:0]  n_e, n_e2;
    reg [22:0] n_frac;
    reg [31:0] n_p;

    always @(*) begin
        if (s3_prod[47]) begin
            n_m  = s3_prod[47:24];
            n_g  = s3_prod[23];
            n_st = |s3_prod[22:0];
            n_e  = s3_exp - 10'd126;       // -127 + 1
        end
        else begin
            n_m  = s3_prod[46:23];
            n_g  = s3_prod[22];
            n_st = |s3_prod[21:0];
            n_e  = s3_exp - 10'd127;
        end

        n_up = n_g & (n_st | n_m[0]);
        n_mr = {1'b0, n_m} + {24'd0, n_up};

        if (n_mr[24]) begin
            n_frac = 23'd0;
            n_e2   = n_e + 10'd1;
        end
        else begin
            n_frac = n_mr[22:0];
            n_e2   = n_e;
        end

        if (s3_zero || n_e2[9] || (n_e2 == 10'd0))
            n_p = 32'd0;                                   // zero / underflow
        else if (n_e2 >= 10'd255)
            n_p = {s3_sign, 8'hFF, 23'd0};                 // overflow -> Inf
        else
            n_p = {s3_sign, n_e2[7:0], n_frac};
    end

    reg [31:0] p_reg;
    always @(posedge clk) begin
        if (clr) p_reg <= 32'd0;
        else     p_reg <= n_p;
    end

    assign p = p_reg;

endmodule

//=====================================================================
// FP32 ADDER  (latency 6)
//   S1 compare/swap  S2 align  S3 add/sub  S4 leading-zero count
//   S5 normalise     S6 round/pack
//=====================================================================
module fp32_add (
    input  wire        clk,
    input  wire        clr,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] s
);

    // ---- stage 1 : unpack, flush denormals, order by magnitude
    wire [7:0]  ea = a[30:23];
    wire [7:0]  eb = b[30:23];
    wire [22:0] fa = (ea == 8'd0) ? 23'd0 : a[22:0];
    wire [22:0] fb = (eb == 8'd0) ? 23'd0 : b[22:0];
    wire        sa = (ea == 8'd0) ? 1'b0  : a[31];
    wire        sb = (eb == 8'd0) ? 1'b0  : b[31];
    wire        a_ge = ({ea, fa} >= {eb, fb});

    reg        s1_sbig, s1_eff_sub;
    reg [7:0]  s1_ebig, s1_diff;
    reg [23:0] s1_mbig, s1_msmall;

    always @(posedge clk) begin
        if (clr) begin
            s1_sbig    <= 1'b0;
            s1_eff_sub <= 1'b0;
            s1_ebig    <= 8'd0;
            s1_diff    <= 8'd0;
            s1_mbig    <= 24'd0;
            s1_msmall  <= 24'd0;
        end
        else begin
            s1_sbig    <= a_ge ? sa : sb;
            s1_eff_sub <= sa ^ sb;
            s1_ebig    <= a_ge ? ea : eb;
            s1_diff    <= a_ge ? (ea - eb) : (eb - ea);
            s1_mbig    <= a_ge ? {(ea != 8'd0), fa} : {(eb != 8'd0), fb};
            s1_msmall  <= a_ge ? {(eb != 8'd0), fb} : {(ea != 8'd0), fa};
        end
    end

    // ---- stage 2 : align the smaller operand (guard/round/sticky = 3 bits)
    wire [26:0] sm = {s1_msmall, 3'b000};
    reg  [26:0] sh;
    reg         stk;

    always @(*) begin
        if (s1_diff >= 8'd27) begin
            sh  = 27'd0;
            stk = |s1_msmall;
        end
        else begin
            sh  = sm >> s1_diff[4:0];
            stk = |(sm & ((27'd1 << s1_diff[4:0]) - 27'd1));
        end
    end

    reg        s2_sbig, s2_eff_sub;
    reg [7:0]  s2_ebig;
    reg [26:0] s2_big, s2_small;

    always @(posedge clk) begin
        if (clr) begin
            s2_sbig    <= 1'b0;
            s2_eff_sub <= 1'b0;
            s2_ebig    <= 8'd0;
            s2_big     <= 27'd0;
            s2_small   <= 27'd0;
        end
        else begin
            s2_sbig    <= s1_sbig;
            s2_eff_sub <= s1_eff_sub;
            s2_ebig    <= s1_ebig;
            s2_big     <= {s1_mbig, 3'b000};
            s2_small   <= sh | {26'd0, stk};
        end
    end

    // ---- stage 3 : add / subtract
    reg        s3_sbig;
    reg [7:0]  s3_ebig;
    reg [27:0] s3_sum;

    always @(posedge clk) begin
        if (clr) begin
            s3_sbig <= 1'b0;
            s3_ebig <= 8'd0;
            s3_sum  <= 28'd0;
        end
        else begin
            s3_sbig <= s2_sbig;
            s3_ebig <= s2_ebig;
            s3_sum  <= s2_eff_sub ? ({1'b0, s2_big} - {1'b0, s2_small})
                                  : ({1'b0, s2_big} + {1'b0, s2_small});
        end
    end

    // ---- stage 4 : leading-zero count
    reg [4:0] lz_c;
    integer   li;
    always @(*) begin
        lz_c = 5'd0;
        for (li = 0; li < 27; li = li + 1)
            if (s3_sum[li])
                lz_c = 26 - li;            // highest set bit wins
    end

    reg        s4_sbig, s4_carry, s4_zero;
    reg [7:0]  s4_ebig;
    reg [4:0]  s4_lz;
    reg [27:0] s4_sum;

    always @(posedge clk) begin
        if (clr) begin
            s4_sbig  <= 1'b0;
            s4_carry <= 1'b0;
            s4_zero  <= 1'b1;
            s4_ebig  <= 8'd0;
            s4_lz    <= 5'd0;
            s4_sum   <= 28'd0;
        end
        else begin
            s4_sbig  <= s3_sbig;
            s4_carry <= s3_sum[27];
            s4_zero  <= (s3_sum == 28'd0);
            s4_ebig  <= s3_ebig;
            s4_lz    <= lz_c;
            s4_sum   <= s3_sum;
        end
    end

    // ---- stage 5 : normalise
    reg [26:0] n5_norm;
    reg [9:0]  n5_e;

    always @(*) begin
        if (s4_carry) begin
            n5_norm = {s4_sum[27:2], s4_sum[1] | s4_sum[0]};
            n5_e    = {2'b00, s4_ebig} + 10'd1;
        end
        else begin
            n5_norm = s4_sum[26:0] << s4_lz;
            n5_e    = {2'b00, s4_ebig} - {5'd0, s4_lz};
        end
    end

    reg        s5_sbig, s5_zero;
    reg [26:0] s5_norm;
    reg [9:0]  s5_e;

    always @(posedge clk) begin
        if (clr) begin
            s5_sbig <= 1'b0;
            s5_zero <= 1'b1;
            s5_norm <= 27'd0;
            s5_e    <= 10'd0;
        end
        else begin
            s5_sbig <= s4_sbig;
            s5_zero <= s4_zero;
            s5_norm <= n5_norm;
            s5_e    <= n5_e;
        end
    end

    // ---- stage 6 : round (nearest-even), pack
    reg [23:0] r_m;
    reg        r_g, r_rs, r_up;
    reg [24:0] r_mr;
    reg [9:0]  r_e;
    reg [22:0] r_frac;
    reg [31:0] r_p;

    always @(*) begin
        r_m  = s5_norm[26:3];
        r_g  = s5_norm[2];
        r_rs = s5_norm[1] | s5_norm[0];
        r_up = r_g & (r_rs | r_m[0]);
        r_mr = {1'b0, r_m} + {24'd0, r_up};

        if (r_mr[24]) begin
            r_frac = 23'd0;
            r_e    = s5_e + 10'd1;
        end
        else begin
            r_frac = r_mr[22:0];
            r_e    = s5_e;
        end

        if (s5_zero || r_e[9] || (r_e == 10'd0))
            r_p = 32'd0;
        else if (r_e >= 10'd255)
            r_p = {s5_sbig, 8'hFF, 23'd0};
        else
            r_p = {s5_sbig, r_e[7:0], r_frac};
    end

    reg [31:0] s_reg;
    always @(posedge clk) begin
        if (clr) s_reg <= 32'd0;
        else     s_reg <= r_p;
    end

    assign s = s_reg;

endmodule

//=====================================================================
// PE (FP32)
//   psum_out = psum_in + data_in * weight_reg       (FP32)
//   weight_reg : stationary weight, shifts DOWN while load_w = 1
//   data_out   : feature to the RIGHT, delayed HOP = 7 cycles
//                (psum hop is 6: this is what lets every column pick
//                 its own feature out of the stream)
//   psum_out   : to the RIGHT, PE_LAT = 10 cycles after data_in
//=====================================================================
module pe_fp32 #(
    parameter HOP = 7                  // feature delay to the next column
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        load_w,
    input  wire        compute_en,

    input  wire [31:0] data_in,        // from the left
    input  wire [31:0] weight_in,      // from the top
    input  wire [31:0] psum_in,        // from the left

    output wire [31:0] data_out,       // to the right
    output wire [31:0] weight_out,     // to the bottom
    output wire [31:0] psum_out        // to the right
);

    reg  [31:0] weight_reg;
    wire [31:0] product;

    // pipelines are cleared between passes
    wire clr = rst || !compute_en;

    assign weight_out = weight_reg;

    always @(posedge clk) begin
        if (rst)
            weight_reg <= 32'd0;
        else if (load_w)
            weight_reg <= weight_in;
    end

    // feature hop delay line
    reg [31:0] dly [0:HOP-1];
    integer di;
    always @(posedge clk) begin
        if (clr) begin
            for (di = 0; di < HOP; di = di + 1)
                dly[di] <= 32'd0;
        end
        else begin
            dly[0] <= data_in;
            for (di = 1; di < HOP; di = di + 1)
                dly[di] <= dly[di-1];
        end
    end

    assign data_out = dly[HOP-1];

    fp32_mul u_mul (
        .clk(clk), .clr(clr),
        .a(data_in), .b(weight_reg),
        .p(product)
    );

    fp32_add u_add (
        .clk(clk), .clr(clr),
        .a(psum_in), .b(product),
        .s(psum_out)
    );

endmodule

//=====================================================================
// N x N systolic array
//   feature_data_in : N words, one per ROW    (left edge)
//   weight_data_in  : N words, one per COLUMN (top edge)
//   conv_result     : N psums, one per ROW    (right edge)
//=====================================================================
module systolic_array #(
    parameter N          = 4,
    parameter DATA_WIDTH = 32,
    parameter ACC_WIDTH  = 32,
    parameter HOP        = 7
)(
    input wire clk,
    input wire rst,
    input wire compute_en,
    input wire load_w,

    input wire [(N*DATA_WIDTH)-1:0] feature_data_in,
    input wire [(N*DATA_WIDTH)-1:0] weight_data_in,

    output wire [(N*ACC_WIDTH)-1:0] conv_result
);

    wire [DATA_WIDTH-1:0] feature_bus [0:N-1][0:N];   // [row][col boundary]
    wire [ACC_WIDTH-1:0]  psum_bus    [0:N-1][0:N];   // [row][col boundary]
    wire [DATA_WIDTH-1:0] weight_bus  [0:N][0:N-1];   // [row boundary][col]

    genvar r, c;
    generate
        for (r = 0; r < N; r = r + 1) begin : ROW_GEN
            // left edge
            assign feature_bus[r][0] = feature_data_in[r*DATA_WIDTH +: DATA_WIDTH];
            assign psum_bus[r][0]    = {ACC_WIDTH{1'b0}};
            // right edge
            assign conv_result[r*ACC_WIDTH +: ACC_WIDTH] = psum_bus[r][N];

            for (c = 0; c < N; c = c + 1) begin : COL_GEN
                // top edge
                if (r == 0) begin : TOP_EDGE
                    assign weight_bus[0][c] =
                        weight_data_in[c*DATA_WIDTH +: DATA_WIDTH];
                end

                pe_fp32 #(.HOP(HOP)) u_pe (
                    .clk(clk),
                    .rst(rst),
                    .load_w(load_w),
                    .compute_en(compute_en),
                    .data_in(feature_bus[r][c]),
                    .weight_in(weight_bus[r][c]),
                    .psum_in(psum_bus[r][c]),
                    .data_out(feature_bus[r][c+1]),
                    .weight_out(weight_bus[r+1][c]),
                    .psum_out(psum_bus[r][c+1])
                );
            end
        end
    endgenerate

endmodule

//=====================================================================
// Controller FSM
//   Per pass: WEIGHT_SHIFT (N cycles) -> COMPUTE (last_phase+1 cycles)
//             -> NEXT_TILE.   After the last pass: DRAIN (FP adder
//             pipeline of the output buffer) -> DONE.
//   output_store strobes in every COMPUTE phase from CAP_BASE to
//   last_phase (one result per cycle).
//=====================================================================
module controller_fsm #(
    parameter N            = 4,
    parameter CAP_BASE     = 32,      // phase at which the first result is valid
    parameter DRAIN_CYCLES = 8        // wait after the last pass
)(
    input wire clk,
    input wire rst,
    input wire start,

    input wire input_loaded,
    input wire weight_loaded,
    input wire last_tile,             // from address_generator
    input wire [6:0] last_phase,      // from address_generator (4*L + CAP_BASE - 1)

    output reg  load_input,
    output reg  load_weight,
    output reg  compute_en,
    output reg  output_store,
    output reg  dataflow_load,
    output reg  buf_write_ok,

    output reg [6:0]   phase_cnt,

    output reg  done,                 // sticky until next start
    output wire run_start,            // 1-cycle pulse
    output wire done_pulse,           // 1-cycle pulse
    output wire tile_advance          // 1-cycle pulse per pass
);

    localparam IDLE          = 3'd0;
    localparam LOAD_INPUT    = 3'd1;
    localparam LOAD_WEIGHT   = 3'd2;
    localparam WEIGHT_SHIFT  = 3'd3;
    localparam COMPUTE       = 3'd4;
    localparam NEXT_TILE     = 3'd5;
    localparam DRAIN         = 3'd6;
    localparam DONE_STATE    = 3'd7;

    localparam DRAIN_LAST = DRAIN_CYCLES - 1;

    reg [2:0] state, next_state;
    reg       start_q;

    wire start_edge = start && !start_q;
    assign run_start    = start_edge && (state == IDLE);
    assign done_pulse   = (state == DONE_STATE);
    assign tile_advance = (state == NEXT_TILE);

    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            start_q   <= 1'b0;
            phase_cnt <= 7'd0;
            done      <= 1'b0;
        end
        else begin
            start_q <= start;
            state   <= next_state;

            if (state != next_state)
                phase_cnt <= 7'd0;
            else if (state == WEIGHT_SHIFT || state == COMPUTE || state == DRAIN)
                phase_cnt <= phase_cnt + 1'b1;

            if (run_start)
                done <= 1'b0;
            else if (state == DONE_STATE)
                done <= 1'b1;
        end
    end

    always @(*) begin
        next_state = state;
        case (state)
            IDLE:         if (start_edge)               next_state = LOAD_INPUT;
            LOAD_INPUT:   if (input_loaded)             next_state = LOAD_WEIGHT;
            LOAD_WEIGHT:  if (weight_loaded)            next_state = WEIGHT_SHIFT;
            WEIGHT_SHIFT: if (phase_cnt == N-1)         next_state = COMPUTE;
            COMPUTE:      if (phase_cnt == last_phase)  next_state = NEXT_TILE;
            NEXT_TILE:    next_state = last_tile ? DRAIN : WEIGHT_SHIFT;
            DRAIN:        if (phase_cnt == DRAIN_LAST)  next_state = DONE_STATE;
            DONE_STATE:   next_state = IDLE;
            default:      next_state = IDLE;
        endcase
    end

    always @(*) begin
        load_input    = (state == LOAD_INPUT);
        load_weight   = (state == LOAD_WEIGHT);
        dataflow_load = (state == WEIGHT_SHIFT);
        compute_en    = (state == COMPUTE);

        buf_write_ok  = (state == IDLE)        || (state == LOAD_INPUT) ||
                        (state == LOAD_WEIGHT) || (state == DONE_STATE);

        output_store  = (state == COMPUTE) &&
                        (phase_cnt >= CAP_BASE) && (phase_cnt <= last_phase);
    end

endmodule

//=====================================================================
// Write pointer (one per stream) : sequential write address, loaded flag
//=====================================================================
module write_pointer #(
    parameter AW = 4
)(
    input  wire clk,
    input  wire rst,
    input  wire clear,                 // logical size changed
    input  wire [8:0] expected,        // values per set
    input  wire write_req,
    input  wire write_ok,

    output wire wr_en,
    output wire [AW-1:0] wr_addr,
    output reg  loaded
);

    reg [8:0] count;

    assign wr_en   = write_req && write_ok;
    assign wr_addr = count[AW-1:0];

    always @(posedge clk) begin
        if (rst || clear) begin
            count  <= 9'd0;
            loaded <= 1'b0;
        end
        else if (wr_en) begin
            if (count == 9'd0)
                loaded <= 1'b0;

            if (count == expected - 1'b1) begin
                count  <= 9'd0;
                loaded <= 1'b1;
            end
            else
                count <= count + 1'b1;
        end
    end

endmodule

//=====================================================================
// ADDRESS GENERATOR
//   * write side : sequential row-major write addresses + loaded flags
//                  for the feature buffer (X) and the weight buffer (W),
//                  L*L words each
//   * pass sequencer : r_tile (outer) / c_tile (inner), flags
//                  tile_valid (sequence running), tiles_done, last_tile
//   * read side  (everything derived from the compute phase of the pass)
//       weights  : 4 lane addresses (one per array column) during the
//                  4 WEIGHT_SHIFT cycles
//                  lane c : W[c_tile*4 + c][r_tile*4 + row_sel],
//                  row_sel = 3 - phase      (first word ends in row 3)
//       features : stream address in phases 0 .. 4L-1
//                  phase j -> m = j/4, c = 3 - j%4,  X[m][c_tile*4 + c]
//       outputs  : index in phases CAP_BASE .. 4L+CAP_BASE-1
//                  u = phase - CAP_BASE -> m = u/4, r = u%4,
//                  Y[m][r_tile*4 + r]  (= output buffer address)
//=====================================================================
module address_generator #(
    parameter N        = 4,
    parameter CAP_BASE = 32
)(
    input  wire clk,
    input  wire rst,
    input  wire [1:0] logical_size,       // 0:4  1:8  2:16

    // host write side
    input  wire feature_write,
    input  wire weight_write,
    input  wire buf_write_ok,
    output wire feature_wr_en,
    output wire weight_wr_en,
    output wire [7:0] feature_wr_addr,
    output wire [7:0] weight_wr_addr,
    output wire input_loaded,
    output wire weight_loaded,

    // pass sequencing
    input  wire tile_start,               // pulse: restart the sequence
    input  wire tile_advance,             // pulse: current pass finished
    input  wire compute_en,
    input  wire [6:0] phase_cnt,
    output reg  [1:0] r_tile,             // output-column group (outer)
    output reg  [1:0] c_tile,             // k group             (inner)
    output wire accumulate_enable,        // c_tile != 0
    output wire last_tile,
    output wire tile_valid,
    output wire tiles_done,
    output wire [6:0] last_phase,

    // read addresses
    output wire stream_valid,
    output wire [7:0]  feature_rd_addr,   // X stream address
    output wire [31:0] weight_rd_addr,    // 4 lanes x 8 bit
    output wire [7:0]  out_index,         // Y index of the value captured now
    output wire [1:0]  out_row            // array row holding that value
);

    reg [8:0] exp_vec;                    // L*L
    reg [2:0] l_shift;                    // log2(L)
    reg [6:0] stream_len;                 // 4L
    reg [4:0] total_tiles;
    reg [4:0] tile_counter;
    reg [1:0] size_q;
    reg       tiles_done_r, tile_valid_r;

    wire size_changed = (logical_size != size_q);

    always @(*) begin
        case (logical_size)
            2'b01: begin
                exp_vec     = 9'd64;
                l_shift     = 3'd3;
                stream_len  = 7'd32;
                total_tiles = 5'd4;
                c_tile = {1'b0, tile_counter[0]};
                r_tile = {1'b0, tile_counter[1]};
            end
            2'b10: begin
                exp_vec     = 9'd256;
                l_shift     = 3'd4;
                stream_len  = 7'd64;
                total_tiles = 5'd16;
                c_tile = tile_counter[1:0];
                r_tile = tile_counter[3:2];
            end
            default: begin
                exp_vec     = 9'd16;
                l_shift     = 3'd2;
                stream_len  = 7'd16;
                total_tiles = 5'd1;
                c_tile = 2'd0;
                r_tile = 2'd0;
            end
        endcase
    end

    assign last_phase        = stream_len + CAP_BASE - 1;
    assign last_tile         = (tile_counter == total_tiles - 1'b1);
    assign accumulate_enable = (c_tile != 2'd0);
    assign tile_valid        = tile_valid_r;
    assign tiles_done        = tiles_done_r;

    // ---- feature stream address  X[m][c_tile*4 + c]
    wire [3:0] m_s = phase_cnt[5:2];
    wire [1:0] c_s = ~phase_cnt[1:0];
    wire [8:0] fa  = ({5'd0, m_s} << l_shift) + {5'd0, c_tile, 2'b00} + {7'd0, c_s};
    assign feature_rd_addr = fa[7:0];
    assign stream_valid    = compute_en && (phase_cnt < stream_len);

    // ---- weight lane addresses  W[c_tile*4 + c][r_tile*4 + row_sel]
    wire [1:0] row_sel = ~phase_cnt[1:0];
    genvar gc;
    generate
        for (gc = 0; gc < N; gc = gc + 1) begin : WLANE
            wire [3:0] krow = {c_tile, 2'b00} + gc;
            wire [8:0] wa   = ({5'd0, krow} << l_shift) + {5'd0, r_tile, 2'b00} + {7'd0, row_sel};
            assign weight_rd_addr[gc*8 +: 8] = wa[7:0];
        end
    endgenerate

    // ---- output index  Y[m][r_tile*4 + r]
    wire [6:0] u   = phase_cnt - CAP_BASE;
    wire [3:0] m_o = u[5:2];
    assign out_row   = u[1:0];
    wire [8:0] oa  = ({5'd0, m_o} << l_shift) + {5'd0, r_tile, 2'b00} + {7'd0, out_row};
    assign out_index = oa[7:0];

    // ---- pass counter
    always @(posedge clk) begin
        if (rst) begin
            size_q       <= 2'b00;
            tile_counter <= 5'd0;
            tiles_done_r <= 1'b0;
            tile_valid_r <= 1'b0;
        end
        else begin
            size_q <= logical_size;

            if (tile_start) begin
                tile_counter <= 5'd0;
                tiles_done_r <= 1'b0;
                tile_valid_r <= 1'b1;
            end
            else if (tile_advance) begin
                if (last_tile) begin
                    tiles_done_r <= 1'b1;
                    tile_valid_r <= 1'b0;
                end
                else
                    tile_counter <= tile_counter + 1'b1;
            end
        end
    end

    write_pointer #(.AW(8)) FEATURE_PTR (
        .clk(clk), .rst(rst), .clear(size_changed),
        .expected(exp_vec),
        .write_req(feature_write),
        .write_ok(buf_write_ok),
        .wr_en(feature_wr_en),
        .wr_addr(feature_wr_addr),
        .loaded(input_loaded)
    );

    write_pointer #(.AW(8)) WEIGHT_PTR (
        .clk(clk), .rst(rst), .clear(size_changed),
        .expected(exp_vec),
        .write_req(weight_write),
        .write_ok(buf_write_ok),
        .wr_en(weight_wr_en),
        .wr_addr(weight_wr_addr),
        .loaded(weight_loaded)
    );

endmodule

//=====================================================================
// Vector memory : plain memory, one write port, N independent
// asynchronous read ports  (feature buffer : N = 1, weight buffer : N = 4)
//=====================================================================
module vector_mem #(
    parameter N     = 1,
    parameter DW    = 32,
    parameter DEPTH = 256,
    parameter AW    = 8
)(
    input  wire clk,
    input  wire wr_en,
    input  wire [AW-1:0] wr_addr,
    input  wire [DW-1:0] data_in,
    input  wire [(N*AW)-1:0] rd_addr,
    output wire [(N*DW)-1:0] data_out
);

    reg [DW-1:0] mem [0:DEPTH-1];

    integer k;
    initial begin
        for (k = 0; k < DEPTH; k = k + 1)
            mem[k] = 0;
    end

    always @(posedge clk) begin
        if (wr_en)
            mem[wr_addr] <= data_in;
    end

    genvar gi;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : RD
            assign data_out[gi*DW +: DW] = mem[rd_addr[gi*AW +: AW]];
        end
    endgenerate

endmodule

//=====================================================================
// Output buffer : L*L results (index = m*L + n), FP32 accumulation
//   first c tile  : direct write
//   other c tiles : memory + new value through the pipelined FP32 adder
//                   (6 cycles), written back with a tagged index
//   one capture per cycle, one memory write port, host read port
//=====================================================================
module output_buffer #(
    parameter N         = 4,
    parameter ACC_WIDTH = 32,
    parameter ADD_LAT   = 6
)(
    input wire clk,
    input wire rst,

    input wire        output_store,              // capture strobe
    input wire        accumulate_enable,         // c_tile != 0
    input wire [7:0]  out_index,                 // Y index of the captured value
    input wire [1:0]  out_row,                   // array row holding it

    input wire [(N*ACC_WIDTH)-1:0] systolic_output,

    input  wire [7:0]           out_rd_addr,     // host read port
    output wire [ACC_WIDTH-1:0] out_rd_data
);

    reg [ACC_WIDTH-1:0] output_mem [0:255];

    integer k;
    initial begin
        for (k = 0; k < 256; k = k + 1)
            output_mem[k] = 0;
    end

    // ---- stage A : register the capture
    wire [ACC_WIDTH-1:0] cap_val = systolic_output[out_row*ACC_WIDTH +: ACC_WIDTH];

    reg                  vq, accq;
    reg [7:0]            idxq;
    reg [ACC_WIDTH-1:0]  valq;

    always @(posedge clk) begin
        if (rst) begin
            vq   <= 1'b0;
            accq <= 1'b0;
            idxq <= 8'd0;
            valq <= 0;
        end
        else begin
            vq   <= output_store;
            accq <= accumulate_enable;
            idxq <= out_index;
            valq <= cap_val;
        end
    end

    // ---- stage B : accumulate through the FP32 adder
    wire [31:0] mem_rd = output_mem[idxq];
    wire [31:0] acc_s;

    fp32_add u_acc_add (
        .clk(clk), .clr(rst),
        .a(mem_rd), .b(valq),
        .s(acc_s)
    );

    // ---- tags travel with the adder pipeline
    reg [ADD_LAT-1:0]     tv;
    reg [(8*ADD_LAT)-1:0] ti;

    always @(posedge clk) begin
        if (rst) begin
            tv <= {ADD_LAT{1'b0}};
            ti <= {(8*ADD_LAT){1'b0}};
        end
        else begin
            tv <= {tv[ADD_LAT-2:0], (vq && accq)};
            ti <= {ti[(8*(ADD_LAT-1))-1:0], idxq};
        end
    end

    // ---- single write port (write-back has priority; the two can never
    //      collide because a pass with c_tile = 0 starts > ADD_LAT cycles
    //      after the previous pass has finished)
    wire                 wb_valid     = tv[ADD_LAT-1];
    wire                 direct_valid = vq && !accq;
    wire                 wr_en        = wb_valid || direct_valid;
    wire [7:0]           wr_a         = wb_valid ? ti[(8*ADD_LAT)-1 -: 8] : idxq;
    wire [ACC_WIDTH-1:0] wr_d         = wb_valid ? acc_s : valq;

    always @(posedge clk) begin
        if (wr_en)
            output_mem[wr_a] <= wr_d;
    end

    assign out_rd_data = output_mem[out_rd_addr];

endmodule

//=====================================================================
// PERFORMANCE MONITOR  (cycle counters, unchanged)
//=====================================================================
module performance_monitor #(
    parameter COUNTER_WIDTH = 32
)(
    input clk,
    input rst,

    input start,
    input done,

    input load_input,
    input load_weight,
    input compute_en,
    input output_store,
    input dataflow_load,

    output reg [COUNTER_WIDTH-1:0] total_cycles,
    output reg [COUNTER_WIDTH-1:0] feature_load_cycles,
    output reg [COUNTER_WIDTH-1:0] weight_load_cycles,
    output reg [COUNTER_WIDTH-1:0] compute_cycles,
    output reg [COUNTER_WIDTH-1:0] output_store_cycles,
    output reg [COUNTER_WIDTH-1:0] busy_cycles,
    output reg [COUNTER_WIDTH-1:0] idle_cycles,

    output reg perf_busy,
    output reg perf_done
);

    wire activity = load_input || load_weight || compute_en ||
                    output_store || dataflow_load;

    always @(posedge clk) begin
        if (rst) begin
            total_cycles        <= 0;
            feature_load_cycles <= 0;
            weight_load_cycles  <= 0;
            compute_cycles      <= 0;
            output_store_cycles <= 0;
            busy_cycles         <= 0;
            idle_cycles         <= 0;
            perf_busy           <= 1'b0;
            perf_done           <= 1'b0;
        end
        else begin
            perf_done <= 1'b0;

            if (start && !perf_busy) begin
                total_cycles        <= 0;
                feature_load_cycles <= 0;
                weight_load_cycles  <= 0;
                compute_cycles      <= 0;
                output_store_cycles <= 0;
                busy_cycles         <= 0;
                idle_cycles         <= 0;
                perf_busy           <= 1'b1;
            end
            else if (perf_busy) begin
                total_cycles <= total_cycles + 1'b1;

                if (load_input)    feature_load_cycles <= feature_load_cycles + 1'b1;
                if (load_weight)   weight_load_cycles  <= weight_load_cycles  + 1'b1;
                if (compute_en)    compute_cycles      <= compute_cycles      + 1'b1;
                if (output_store)  output_store_cycles <= output_store_cycles + 1'b1;

                if (activity) busy_cycles <= busy_cycles + 1'b1;
                else          idle_cycles <= idle_cycles + 1'b1;

                if (done) begin
                    perf_busy <= 1'b0;
                    perf_done <= 1'b1;
                end
            end
        end
    end

endmodule

//=====================================================================
// Top level
//   FP32 : DATA_WIDTH = 32, ACC_WIDTH = 32
//   N    : physical array size (keep 4); the logical matrix size
//          (4x4 / 8x8 / 16x16) is selected with logical_size.
//   Host writes L*L features (X) and L*L weights (W), row-major, starts the
//   run and reads L*L results (Y) through out_rd_addr / out_rd_data.
//=====================================================================
module cnn_accelerator_top #(
    parameter N             = 4,       // physical array size (keep 4)
    parameter DATA_WIDTH    = 32,      // FP32
    parameter ACC_WIDTH     = 32,      // FP32
    parameter COUNTER_WIDTH = 32
)(
    input clk,
    input rst,
    input [1:0] logical_size,          // 0:4x4  1:8x8  2:16x16
    input start,

    input feature_write,
    input weight_write,
    input [DATA_WIDTH-1:0] feature_data_in,    // FP32 bit pattern
    input [DATA_WIDTH-1:0] weight_data_in,     // FP32 bit pattern

    input  [7:0]           out_rd_addr,        // result read port (m*L + n)
    output [ACC_WIDTH-1:0] out_rd_data,

    output output_valid,
    output done,
    output input_loaded,
    output weight_loaded,

    output load_input,
    output load_weight,
    output compute_en,
    output output_store,
    output dataflow_load,

    output [COUNTER_WIDTH-1:0] perf_total_cycles,
    output [COUNTER_WIDTH-1:0] perf_feature_load_cycles,
    output [COUNTER_WIDTH-1:0] perf_weight_load_cycles,
    output [COUNTER_WIDTH-1:0] perf_compute_cycles,
    output [COUNTER_WIDTH-1:0] perf_output_store_cycles,
    output [COUNTER_WIDTH-1:0] perf_busy_cycles,
    output [COUNTER_WIDTH-1:0] perf_idle_cycles,

    output perf_busy,
    output perf_done
);

    // ---- pipeline latencies (must match the fp32 modules)
    localparam FP_MUL_LAT   = 4;
    localparam FP_ADD_LAT   = 6;

    localparam HOP          = FP_ADD_LAT + 1;                         // feature hop between columns
    localparam PE_LAT       = FP_MUL_LAT + FP_ADD_LAT;                // feature -> psum_out of one PE
    localparam CAP_BASE     = N + ((N-1) * FP_ADD_LAT) + PE_LAT;      // first result valid (32)
    localparam DRAIN_CYCLES = FP_ADD_LAT + 2;                         // output accumulate pipeline

    wire [1:0]   r_tile, c_tile;
    wire [6:0]   phase_cnt, last_phase;
    wire         accumulate_enable;
    wire         buf_write_ok;
    wire         run_start, done_pulse, tile_advance;
    wire         last_tile, tile_valid, tiles_done;

    wire         feature_wr_en, weight_wr_en;
    wire [7:0]   feature_wr_addr, weight_wr_addr;

    wire         stream_valid;
    wire [7:0]   feature_rd_addr, out_index;
    wire [31:0]  weight_rd_addr;
    wire [1:0]   out_row;

    wire [DATA_WIDTH-1:0]     feature_data_out;     // one word / cycle (stream)
    wire [(N*DATA_WIDTH)-1:0] weight_data_out;      // one word per array column
    wire [(N*ACC_WIDTH)-1:0]  conv_result;

    assign output_valid = done;

    //-----------------------------------------------------------------
    controller_fsm #(
        .N(N),
        .CAP_BASE(CAP_BASE),
        .DRAIN_CYCLES(DRAIN_CYCLES)
    ) CONTROLLER (
        .clk(clk), .rst(rst), .start(start),
        .input_loaded(input_loaded),
        .weight_loaded(weight_loaded),
        .last_tile(last_tile),
        .last_phase(last_phase),
        .load_input(load_input),
        .load_weight(load_weight),
        .compute_en(compute_en),
        .output_store(output_store),
        .dataflow_load(dataflow_load),
        .buf_write_ok(buf_write_ok),
        .phase_cnt(phase_cnt),
        .done(done),
        .run_start(run_start),
        .done_pulse(done_pulse),
        .tile_advance(tile_advance)
    );

    address_generator #(.N(N), .CAP_BASE(CAP_BASE)) ADDRESS_GENERATOR (
        .clk(clk), .rst(rst),
        .logical_size(logical_size),
        .feature_write(feature_write),
        .weight_write(weight_write),
        .buf_write_ok(buf_write_ok),
        .feature_wr_en(feature_wr_en),
        .weight_wr_en(weight_wr_en),
        .feature_wr_addr(feature_wr_addr),
        .weight_wr_addr(weight_wr_addr),
        .input_loaded(input_loaded),
        .weight_loaded(weight_loaded),
        .tile_start(run_start),
        .tile_advance(tile_advance),
        .compute_en(compute_en),
        .phase_cnt(phase_cnt),
        .r_tile(r_tile),
        .c_tile(c_tile),
        .accumulate_enable(accumulate_enable),
        .last_tile(last_tile),
        .tile_valid(tile_valid),
        .tiles_done(tiles_done),
        .last_phase(last_phase),
        .stream_valid(stream_valid),
        .feature_rd_addr(feature_rd_addr),
        .weight_rd_addr(weight_rd_addr),
        .out_index(out_index),
        .out_row(out_row)
    );

    // X : L*L features, one read port (the serial feature stream)
    vector_mem #(
        .N(1), .DW(DATA_WIDTH), .DEPTH(256), .AW(8)
    ) INPUT_BUFFER (
        .clk(clk),
        .wr_en(feature_wr_en),
        .wr_addr(feature_wr_addr),
        .data_in(feature_data_in),
        .rd_addr(feature_rd_addr),
        .data_out(feature_data_out)
    );

    // W : L*L weights, 4 read ports (one per array COLUMN); the 4 words of
    // a shift cycle go down the array, 4 shift cycles fill the 4x4 block
    vector_mem #(
        .N(N), .DW(DATA_WIDTH), .DEPTH(256), .AW(8)
    ) WEIGHT_BUFFER (
        .clk(clk),
        .wr_en(weight_wr_en),
        .wr_addr(weight_wr_addr),
        .data_in(weight_data_in),
        .rd_addr(weight_rd_addr),
        .data_out(weight_data_out)
    );

    // ---- feature stream register + row delay lines
    //      array row 0 gets the stream, row r the stream delayed r more
    //      cycles, so the four row results leave on consecutive cycles
    wire clr_stream = rst || !compute_en;
    reg [DATA_WIDTH-1:0] stream_r, fd1, fd2, fd3;

    always @(posedge clk) begin
        if (clr_stream) begin
            stream_r <= 0;
            fd1      <= 0;
            fd2      <= 0;
            fd3      <= 0;
        end
        else begin
            stream_r <= stream_valid ? feature_data_out : {DATA_WIDTH{1'b0}};
            fd1      <= stream_r;
            fd2      <= fd1;
            fd3      <= fd2;
        end
    end

    wire [(N*DATA_WIDTH)-1:0] feature_skewed = {fd3, fd2, fd1, stream_r};

    systolic_array #(
        .N(N), .DATA_WIDTH(DATA_WIDTH), .ACC_WIDTH(ACC_WIDTH), .HOP(HOP)
    ) SYSTOLIC_ARRAY (
        .clk(clk), .rst(rst),
        .compute_en(compute_en),
        .load_w(dataflow_load),
        .feature_data_in(feature_skewed),
        .weight_data_in(weight_data_out),
        .conv_result(conv_result)
    );

    output_buffer #(
        .N(N), .ACC_WIDTH(ACC_WIDTH), .ADD_LAT(FP_ADD_LAT)
    ) OUTPUT_BUFFER (
        .clk(clk), .rst(rst),
        .output_store(output_store),
        .accumulate_enable(accumulate_enable),
        .out_index(out_index),
        .out_row(out_row),
        .systolic_output(conv_result),
        .out_rd_addr(out_rd_addr),
        .out_rd_data(out_rd_data)
    );

    performance_monitor #(.COUNTER_WIDTH(COUNTER_WIDTH)) PERFORMANCE_MONITOR (
        .clk(clk), .rst(rst),
        .start(run_start),
        .done(done_pulse),
        .load_input(load_input),
        .load_weight(load_weight),
        .compute_en(compute_en),
        .output_store(output_store),
        .dataflow_load(dataflow_load),
        .total_cycles(perf_total_cycles),
        .feature_load_cycles(perf_feature_load_cycles),
        .weight_load_cycles(perf_weight_load_cycles),
        .compute_cycles(perf_compute_cycles),
        .output_store_cycles(perf_output_store_cycles),
        .busy_cycles(perf_busy_cycles),
        .idle_cycles(perf_idle_cycles),
        .perf_busy(perf_busy),
        .perf_done(perf_done)
    );

endmodule