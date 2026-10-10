`timescale 1ns/1ps
// ============================================================
// keccak_f200_layer.v - C2 obfuscation layer (Section 3.4)
//
// Collects 64 raw response bits, XORs them into lanes 0..7 of the
// 200-bit Keccak-f[200] state (bit i of the block -> lane i/8,
// bit i%8), applies the 18-round permutation (one round per
// clock), and then emits lanes 0..7 of the new state as 64
// obfuscated bits, bit 0 first. The state is kept between blocks.
// Bit-exact with KeccakF200Model in generate_c1_c2_datasets.py
// (checked by tb_keccak_layer.v + tools/check_keccak.py).
// ============================================================
module keccak_f200_layer (
    input  wire clk,
    input  wire rst_n,
    input  wire in_bit,
    input  wire in_valid,
    output reg  out_bit,
    output reg  out_valid
);
    reg [199:0] st;
    reg [63:0]  acc;  reg [6:0] acc_n;
    reg [63:0]  obuf; reg [6:0] out_n;
    reg [4:0]   rnd;
    reg         perm, emit;
    wire [199:0] st_next;

    keccak_f200_round u_round (.s_in(st), .rnd(rnd), .s_out(st_next));

    wire [63:0] acc_full = {in_bit, acc[63:1]};     // block bit i ends at acc[i]

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= 200'd0; acc <= 64'd0; acc_n <= 7'd0; obuf <= 64'd0; out_n <= 7'd0;
            rnd <= 5'd0; perm <= 1'b0; emit <= 1'b0; out_bit <= 1'b0; out_valid <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            if (in_valid) begin
                if (acc_n == 7'd63) begin           // block complete: absorb and start
                    st[63:0] <= st[63:0] ^ acc_full;
                    acc_n <= 7'd0; perm <= 1'b1; rnd <= 5'd0;
                end else begin
                    acc <= acc_full; acc_n <= acc_n + 1;
                end
            end
            if (perm) begin
                st  <= st_next;
                rnd <= rnd + 1;
                if (rnd == 5'd17) begin
                    perm <= 1'b0; emit <= 1'b1; out_n <= 7'd0;
                    obuf <= st_next[63:0];
                end
            end
            if (emit) begin
                out_bit <= obuf[0]; out_valid <= 1'b1;
                obuf <= {1'b0, obuf[63:1]}; out_n <= out_n + 1;
                if (out_n == 7'd63) emit <= 1'b0;
            end
        end
    end
endmodule

// ---------------- one round of Keccak-f[200] (8-bit lanes) ----------------
module keccak_f200_round (
    input  wire [199:0] s_in,
    input  wire [4:0]   rnd,
    output wire [199:0] s_out
);
    function integer rho_of;  input integer i;
        case (i)
            0:rho_of=0;  1:rho_of=1;  2:rho_of=6;  3:rho_of=4;  4:rho_of=3;
            5:rho_of=4;  6:rho_of=4;  7:rho_of=6;  8:rho_of=7;  9:rho_of=4;
           10:rho_of=3; 11:rho_of=2; 12:rho_of=3; 13:rho_of=1; 14:rho_of=7;
           15:rho_of=1; 16:rho_of=5; 17:rho_of=7; 18:rho_of=5; 19:rho_of=0;
           20:rho_of=2; 21:rho_of=2; 22:rho_of=5; 23:rho_of=0; default:rho_of=6;
        endcase
    endfunction
    function integer pi_of;  input integer i;
        case (i)
            0:pi_of=0;   1:pi_of=15;  2:pi_of=20;  3:pi_of=10;  4:pi_of=5;
            5:pi_of=1;   6:pi_of=16;  7:pi_of=21;  8:pi_of=11;  9:pi_of=6;
           10:pi_of=2;  11:pi_of=17; 12:pi_of=22; 13:pi_of=12; 14:pi_of=7;
           15:pi_of=3;  16:pi_of=18; 17:pi_of=23; 18:pi_of=13; 19:pi_of=8;
           20:pi_of=4;  21:pi_of=19; 22:pi_of=24; 23:pi_of=14; default:pi_of=9;
        endcase
    endfunction

    reg [7:0] rc;
    always @(*) case (rnd)
        5'd0:rc=8'h01;  5'd1:rc=8'h82;  5'd2:rc=8'h8A;  5'd3:rc=8'h00;  5'd4:rc=8'h8B;
        5'd5:rc=8'h01;  5'd6:rc=8'h81;  5'd7:rc=8'h09;  5'd8:rc=8'h8A;  5'd9:rc=8'h88;
        5'd10:rc=8'h09; 5'd11:rc=8'h0A; 5'd12:rc=8'h8B; 5'd13:rc=8'h8B; 5'd14:rc=8'h89;
        5'd15:rc=8'h03; 5'd16:rc=8'h02; default:rc=8'h80;
    endcase

    wire [7:0] C [0:4];
    wire [7:0] D [0:4];
    wire [7:0] th [0:24];
    wire [7:0] tp [0:24];
    genvar x, y, k;
    generate
        for (x = 0; x < 5; x = x + 1) begin : g_c
            assign C[x] = s_in[8*x +: 8] ^ s_in[8*(5+x) +: 8] ^ s_in[8*(10+x) +: 8]
                        ^ s_in[8*(15+x) +: 8] ^ s_in[8*(20+x) +: 8];
        end
        for (x = 0; x < 5; x = x + 1) begin : g_d
            assign D[x] = C[(x+4)%5] ^ {C[(x+1)%5][6:0], C[(x+1)%5][7]};
        end
        for (k = 0; k < 25; k = k + 1) begin : g_th
            assign th[k] = s_in[8*k +: 8] ^ D[k%5];
            // rho + pi: tp[pi(k)] = rotl(th[k], rho(k))
            assign tp[pi_of(k)] = (th[k] << rho_of(k)) | (th[k] >> (8 - rho_of(k)));
        end
        for (y = 0; y < 5; y = y + 1) begin : g_y
            for (x = 0; x < 5; x = x + 1) begin : g_x
                if (x == 0 && y == 0) begin : g_iota
                    assign s_out[7:0] = (tp[0] ^ (~tp[1] & tp[2])) ^ rc;
                end else begin : g_chi
                    assign s_out[8*(5*y+x) +: 8] = tp[5*y+x] ^ (~tp[5*y+(x+1)%5] & tp[5*y+(x+2)%5]);
                end
            end
        end
    endgenerate
endmodule
