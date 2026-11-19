
`timescale 1ns/1ps

module tb_cnn_accelerator_top;

    // Clock and control
    reg clk = 0;
    always #5 clk = ~clk;

    reg rst, start;
    reg [1:0] logical_size;
    reg feature_write, weight_write;
    reg [31:0] feature_data_in, weight_data_in;
    reg [7:0] out_rd_addr;

    // Outputs
    wire [31:0] out_rd_data;
    wire done, input_loaded, weight_loaded;
    wire load_input, load_weight, compute_en;
    wire output_store, dataflow_load;

    wire [31:0] perf_total_cycles;
    wire [31:0] perf_compute_cycles;
    wire [31:0] perf_busy_cycles;
    wire [31:0] perf_idle_cycles;

    // Internal FSM and dataflow signals for waveform viewing
    wire [2:0] fsm_state;
    wire [6:0] phase_count;
    wire [1:0] row_tile, col_tile;
    wire [31:0] systolic_result;

    integer i, timeout_count;
    reg [31:0] result [0:15];
    reg [2:0] previous_state;

    // Connect internal DUT signals to visible testbench wires
    // These hierarchical references are for simulation/debugging.
    cnn_accelerator_top #(
        .N(4),
        .DATA_WIDTH(32),
        .ACC_WIDTH(32)
    ) DUT (
        .clk(clk),
        .rst(rst),
        .logical_size(logical_size),
        .start(start),
        .feature_write(feature_write),
        .weight_write(weight_write),
        .feature_data_in(feature_data_in),
        .weight_data_in(weight_data_in),
        .out_rd_addr(out_rd_addr),
        .out_rd_data(out_rd_data),
        .done(done),
        .input_loaded(input_loaded),
        .weight_loaded(weight_loaded),
        .load_input(load_input),
        .load_weight(load_weight),
        .compute_en(compute_en),
        .output_store(output_store),
        .dataflow_load(dataflow_load),
        .perf_total_cycles(perf_total_cycles),
        .perf_compute_cycles(perf_compute_cycles),
        .perf_busy_cycles(perf_busy_cycles),
        .perf_idle_cycles(perf_idle_cycles)
    );

    assign fsm_state      = DUT.CONTROLLER.state;
    assign phase_count    = DUT.phase_cnt;
    assign row_tile       = DUT.r_tile;
    assign col_tile       = DUT.c_tile;
    assign systolic_result = DUT.conv_result;

    // IEEE-754 FP32 values for 1 through 16
    function [31:0] fp32;
        input integer n;
        begin
            case (n)
                 1: fp32=32'h3F800000;
                 2: fp32=32'h40000000;
                 3: fp32=32'h40400000;
                 4: fp32=32'h40800000;
                 5: fp32=32'h40A00000;
                 6: fp32=32'h40C00000;
                 7: fp32=32'h40E00000;
                 8: fp32=32'h41000000;
                 9: fp32=32'h41100000;
                10: fp32=32'h41200000;
                11: fp32=32'h41300000;
                12: fp32=32'h41400000;
                13: fp32=32'h41500000;
                14: fp32=32'h41600000;
                15: fp32=32'h41700000;
                16: fp32=32'h41800000;
                default: fp32=32'h00000000;
            endcase
        end
    endfunction

    // Sequential row-major feature write
    task write_feature;
        input [31:0] value;
        begin
            @(negedge clk);
            feature_data_in = value;
            feature_write = 1;
            @(negedge clk);
            feature_write = 0;
        end
    endtask

    // Sequential row-major weight write
    task write_weight;
        input [31:0] value;
        begin
            @(negedge clk);
            weight_data_in = value;
            weight_write = 1;
            @(negedge clk);
            weight_write = 0;
        end
    endtask

    // Read one output word
    task read_result;
        input [7:0] address;
        begin
            @(negedge clk);
            out_rd_addr = address;
            #1;
            result[address] = out_rd_data;
        end
    endtask

    // Print FSM transitions for the simulation transcript
    initial previous_state = 3'd0;

    always @(negedge clk) begin
        if (!rst && fsm_state != previous_state) begin
            $display("FSM: state=%0d phase=%0d",
                     fsm_state, phase_count);
            previous_state = fsm_state;
        end
    end

    initial begin
        rst = 1;
        start = 0;
        logical_size = 0;  // 4x4
        feature_write = 0;
        weight_write = 0;
        feature_data_in = 0;
        weight_data_in = 0;
        out_rd_addr = 0;

        repeat (5) @(negedge clk);
        rst = 0;

        // X = 1, 2, 3, ... 16
        for (i=1; i<=16; i=i+1)
            write_feature(fp32(i));

        // W = identity matrix
        // 1 0 0 0
        // 0 1 0 0
        // 0 0 1 0
        // 0 0 0 1
        for (i=0; i<16; i=i+1) begin
            if ((i/4) == (i%4))
                write_weight(fp32(1));
            else
                write_weight(32'h00000000);
        end

        wait (input_loaded && weight_loaded);
        repeat (2) @(negedge clk);

        $display("Starting 4x4 matrix multiplication");
        start = 1;
        @(negedge clk);
        start = 0;

        timeout_count = 0;
        while (done !== 1'b1 && timeout_count < 1000) begin
            @(negedge clk);
            timeout_count = timeout_count + 1;
        end

        if (done !== 1'b1) begin
            $display("ERROR: Accelerator timed out");
        end
        else begin
            $display("Completed. Total cycles = %0d",
                     perf_total_cycles);
            $display("Compute cycles = %0d",
                     perf_compute_cycles);

            for (i=0; i<16; i=i+1)
                read_result(i);

            $display("Result matrix Y (FP32 hexadecimal):");
            for (i=0; i<16; i=i+1) begin
                $write("%h ", result[i]);
                if ((i%4)==3)
                    $write("\n");
            end
        end

        $display("Expected Y = X because W is identity");
        $finish;
    end

    initial begin
        #200000;
        $display("GLOBAL TIMEOUT");
        $finish;
    end

endmodule
