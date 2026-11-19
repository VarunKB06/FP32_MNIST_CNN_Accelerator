
`timescale 1ns / 1ps

module cnn_accelerator_axi #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 11
)(
    input  wire                                  S_AXI_ACLK,
    input  wire                                  S_AXI_ARESETN,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]         S_AXI_AWADDR,
    input  wire [2:0]                            S_AXI_AWPROT,
    input  wire                                  S_AXI_AWVALID,
    output wire                                  S_AXI_AWREADY,

    input  wire [C_S_AXI_DATA_WIDTH-1:0]         S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0]     S_AXI_WSTRB,
    input  wire                                  S_AXI_WVALID,
    output wire                                  S_AXI_WREADY,

    output wire [1:0]                            S_AXI_BRESP,
    output reg                                   S_AXI_BVALID,
    input  wire                                  S_AXI_BREADY,

    input  wire [C_S_AXI_ADDR_WIDTH-1:0]         S_AXI_ARADDR,
    input  wire [2:0]                            S_AXI_ARPROT,
    input  wire                                  S_AXI_ARVALID,
    output wire                                  S_AXI_ARREADY,

    output reg  [C_S_AXI_DATA_WIDTH-1:0]         S_AXI_RDATA,
    output wire [1:0]                            S_AXI_RRESP,
    output reg                                   S_AXI_RVALID,
    input  wire                                  S_AXI_RREADY
);

    localparam [31:0] ID_VALUE = 32'h4D4D3031;

    //-----------------------------------------------------------------
    // AXI4-Lite WRITE CHANNEL
    // AW and W channels are accepted independently.
    //-----------------------------------------------------------------

    reg [C_S_AXI_ADDR_WIDTH-1:0] awaddr_reg;
    reg [31:0] wdata_reg;

    reg aw_pending;
    reg w_pending;

    wire aw_handshake = S_AXI_AWVALID && S_AXI_AWREADY;
    wire w_handshake  = S_AXI_WVALID  && S_AXI_WREADY;

    assign S_AXI_AWREADY = !aw_pending && !S_AXI_BVALID;
    assign S_AXI_WREADY  = !w_pending  && !S_AXI_BVALID;

    wire wr_commit =
        !S_AXI_BVALID &&
        (aw_pending || aw_handshake) &&
        (w_pending  || w_handshake);

    wire [C_S_AXI_ADDR_WIDTH-1:0] wr_addr =
        aw_pending ? awaddr_reg : S_AXI_AWADDR;

    wire [31:0] wr_data =
        w_pending ? wdata_reg : S_AXI_WDATA;

    wire [8:0] wr_idx = wr_addr[10:2];

    assign S_AXI_BRESP = 2'b00;
    assign S_AXI_RRESP = 2'b00;

    //-----------------------------------------------------------------
    // CORE CONTROL SIGNALS
    //-----------------------------------------------------------------

    reg start_r;
    reg swrst_r;

    reg feat_wr_r;
    reg wt_wr_r;

    reg [31:0] feat_data_r;
    reg [31:0] wt_data_r;

    reg [1:0] size_r;

    //-----------------------------------------------------------------
    // WRITE CHANNEL REGISTERS
    //-----------------------------------------------------------------

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin

            awaddr_reg <= {C_S_AXI_ADDR_WIDTH{1'b0}};
            wdata_reg  <= 32'd0;

            aw_pending <= 1'b0;
            w_pending  <= 1'b0;

            S_AXI_BVALID <= 1'b0;

            start_r     <= 1'b0;
            swrst_r     <= 1'b0;

            feat_wr_r   <= 1'b0;
            wt_wr_r     <= 1'b0;

            feat_data_r <= 32'd0;
            wt_data_r   <= 32'd0;

            size_r      <= 2'd0;

        end
        else begin

            // Default one-cycle pulses.
            start_r   <= 1'b0;
            swrst_r   <= 1'b0;

            feat_wr_r <= 1'b0;
            wt_wr_r   <= 1'b0;

            // Capture address independently.
            if (aw_handshake) begin
                awaddr_reg <= S_AXI_AWADDR;
                aw_pending <= 1'b1;
            end

            // Capture write data independently.
            if (w_handshake) begin
                wdata_reg  <= S_AXI_WDATA;
                w_pending  <= 1'b1;
            end

            // Commit once both address and data are available.
            if (wr_commit) begin

                aw_pending <= 1'b0;
                w_pending  <= 1'b0;

                S_AXI_BVALID <= 1'b1;

                case (wr_idx)

                    9'h000: begin
                        // CTRL
                        start_r <= wr_data[0];
                        swrst_r <= wr_data[1];
                    end

                    9'h001: begin
                        // SIZE
                        size_r <= wr_data[1:0];
                    end

                    9'h003: begin
                        // FEATURE
                        feat_data_r <= wr_data;
                        feat_wr_r   <= 1'b1;
                    end

                    9'h004: begin
                        // WEIGHT
                        wt_data_r <= wr_data;
                        wt_wr_r   <= 1'b1;
                    end

                    default: begin
                        // Read-only or unmapped address.
                    end

                endcase
            end

            // AXI write response handshake.
            if (S_AXI_BVALID && S_AXI_BREADY) begin
                S_AXI_BVALID <= 1'b0;
            end

        end
    end

    //-----------------------------------------------------------------
    // ACCELERATOR CORE
    //-----------------------------------------------------------------

    wire core_rst = !S_AXI_ARESETN || swrst_r;

    wire [7:0] out_rd_addr;
    wire [31:0] out_rd_data;

    wire output_valid;
    wire done;
    wire input_loaded;
    wire weight_loaded;

    wire load_input;
    wire load_weight;
    wire compute_en;
    wire output_store;
    wire dataflow_load;

    wire [31:0] perf_total_cycles;
    wire [31:0] perf_feature_load_cycles;
    wire [31:0] perf_weight_load_cycles;
    wire [31:0] perf_compute_cycles;
    wire [31:0] perf_output_store_cycles;
    wire [31:0] perf_busy_cycles;
    wire [31:0] perf_idle_cycles;

    wire perf_busy;
    wire perf_done;

    cnn_accelerator_top #(
        .N(4),
        .DATA_WIDTH(32),
        .ACC_WIDTH(32),
        .COUNTER_WIDTH(32)
    ) u_core (

        .clk(S_AXI_ACLK),
        .rst(core_rst),

        .logical_size(size_r),
        .start(start_r),

        .feature_write(feat_wr_r),
        .weight_write(wt_wr_r),

        .feature_data_in(feat_data_r),
        .weight_data_in(wt_data_r),

        .out_rd_addr(out_rd_addr),
        .out_rd_data(out_rd_data),

        .output_valid(output_valid),
        .done(done),

        .input_loaded(input_loaded),
        .weight_loaded(weight_loaded),

        .load_input(load_input),
        .load_weight(load_weight),
        .compute_en(compute_en),
        .output_store(output_store),
        .dataflow_load(dataflow_load),

        .perf_total_cycles(perf_total_cycles),
        .perf_feature_load_cycles(perf_feature_load_cycles),
        .perf_weight_load_cycles(perf_weight_load_cycles),
        .perf_compute_cycles(perf_compute_cycles),
        .perf_output_store_cycles(perf_output_store_cycles),
        .perf_busy_cycles(perf_busy_cycles),
        .perf_idle_cycles(perf_idle_cycles),

        .perf_busy(perf_busy),
        .perf_done(perf_done)
    );

    //-----------------------------------------------------------------
    // AXI4-Lite READ CHANNEL
    //-----------------------------------------------------------------

    assign S_AXI_ARREADY = !S_AXI_RVALID;

    wire rd_fire = S_AXI_ARVALID && S_AXI_ARREADY;

    wire [8:0] rd_idx = S_AXI_ARADDR[10:2];

    // Output memory window: 0x400 to 0x7FC.
    wire out_win = S_AXI_ARADDR[10];

    assign out_rd_addr = S_AXI_ARADDR[9:2];

    reg [31:0] rd_mux;

    always @(*) begin

        rd_mux = 32'd0;

        if (out_win) begin

            rd_mux = out_rd_data;

        end
        else begin

            case (rd_idx)

                9'h001:
                    rd_mux = {30'd0, size_r};

                9'h002:
                    rd_mux = {
                        27'd0,
                        output_valid,
                        weight_loaded,
                        input_loaded,
                        perf_busy,
                        done
                    };

                9'h005:
                    rd_mux = perf_total_cycles;

                9'h006:
                    rd_mux = perf_feature_load_cycles;

                9'h007:
                    rd_mux = perf_weight_load_cycles;

                9'h008:
                    rd_mux = perf_compute_cycles;

                9'h009:
                    rd_mux = perf_output_store_cycles;

                9'h00A:
                    rd_mux = perf_busy_cycles;

                9'h00B:
                    rd_mux = perf_idle_cycles;

                9'h00C:
                    rd_mux = ID_VALUE;

                default:
                    rd_mux = 32'd0;

            endcase

        end
    end

    //-----------------------------------------------------------------
    // READ RESPONSE
    //-----------------------------------------------------------------

    always @(posedge S_AXI_ACLK) begin

        if (!S_AXI_ARESETN) begin

            S_AXI_RVALID <= 1'b0;
            S_AXI_RDATA  <= 32'd0;

        end
        else begin

            if (rd_fire) begin

                S_AXI_RVALID <= 1'b1;
                S_AXI_RDATA  <= rd_mux;

            end
            else if (S_AXI_RVALID && S_AXI_RREADY) begin

                S_AXI_RVALID <= 1'b0;

            end

        end
    end

endmodule
