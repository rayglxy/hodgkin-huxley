% hodgkin_huxley.m
% ----------------
% MATLAB implementation of the Hodgkin-Huxley (1952) neuron model.
% Parallel implementation to the Python version in src/hodgkin_huxley.py
%
% Reference:
%   Hodgkin & Huxley (1952). A quantitative description of membrane current
%   and its application to conduction and excitation in nerve.
%   Journal of Physiology, 117, 500-544.
%
% Usage:
%   Run this script directly to produce all three figures.

clear; clc; close all;

%% --- Parameters ---
C_m  = 1.0;    % membrane capacitance (uF/cm^2)
g_Na = 120.0;  % max Na conductance (mS/cm^2)
g_K  = 36.0;   % max K conductance (mS/cm^2)
g_L  = 0.3;    % leak conductance (mS/cm^2)
E_Na = 115.0;  % Na reversal potential (mV)
E_K  = -12.0;  % K reversal potential (mV)
E_L  = 10.6;   % leak reversal potential (mV)

params = [C_m, g_Na, g_K, g_L, E_Na, E_K, E_L];

%% --- Initial conditions (resting state) ---
V0 = 0.0;
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0));
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0));
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0));
state0 = [V0, m0, h0, n0];

%% ============================================================
%% FIGURE 1 — Action potential at I_ext = 10 uA/cm^2
%% ============================================================
I_ext = 10.0;
tspan = [0 50];
opts  = odeset('MaxStep', 0.05);

[t, y] = ode45(@(t,y) hh_odes(t, y, I_ext, params), tspan, state0, opts);

figure(1);
plot(t, y(:,1), 'Color', [0.33 0.29 0.72], 'LineWidth', 1.8);
xlabel('Time (ms)');
ylabel('Membrane voltage (mV)');
title(sprintf('Action potential — I_{ext} = %.1f \\muA/cm^2', I_ext));
box off;
saveas(gcf, '../figures/action_potential_matlab.png');

%% ============================================================
%% FIGURE 2 — f-I curve and bifurcation
%% ============================================================
I_values     = 0:0.5:50;
firing_rates = zeros(size(I_values));

for k = 1:length(I_values)
    I = I_values(k);
    [~, y_sim] = ode45(@(t,y) hh_odes(t, y, I, params), ...
                       [0 200], state0, opts);
    firing_rates(k) = count_spikes(y_sim(:,1), 50.0) / 200 * 1000;
end

% find bifurcation point
bif_idx = find(firing_rates > 0, 1);
I_bif   = I_values(bif_idx);

figure(2);
plot(I_values, firing_rates, 'Color', [0.33 0.29 0.72], 'LineWidth', 2);
hold on;
xline(I_bif, '--', sprintf('bifurcation at I = %.1f', I_bif), ...
      'Color', [0.85 0.35 0.19], 'LineWidth', 1.5);
xlabel('Injected current I (\muA/cm^2)');
ylabel('Firing rate (Hz)');
title('f-I curve — firing rate vs injected current');
box off;
saveas(gcf, '../figures/fi_curve_matlab.png');

%% ============================================================
%% FIGURE 3 — Phase portrait
%% ============================================================
[t3, y3] = ode45(@(t,y) hh_odes(t, y, 10.0, params), [0 300], state0, opts);

% discard first 100ms transient
mask = t3 > 100;
V_plot = y3(mask, 1);
n_plot = y3(mask, 4);

% nullclines
V_range    = linspace(-20, 120, 500);
n_nullcline = alpha_n(V_range) ./ (alpha_n(V_range) + beta_n(V_range));

m_ss = alpha_m(V_range) ./ (alpha_m(V_range) + beta_m(V_range));
h_ss = alpha_h(V_range) ./ (alpha_h(V_range) + beta_h(V_range));
I_Na_ss = g_Na .* m_ss.^3 .* h_ss .* (V_range - E_Na);
I_L_ss  = g_L .* (V_range - E_L);
n4      = (10.0 - I_Na_ss - I_L_ss) ./ (g_K .* (V_range - E_K));
valid   = (n4 >= 0) & (n4 <= 1);

figure(3);
plot(V_plot, n_plot, 'Color', [0.33 0.29 0.72], 'LineWidth', 1.8); hold on;
plot(V_range, n_nullcline, '--', 'Color', [0.11 0.62 0.46], 'LineWidth', 1.5);
plot(V_range(valid), n4(valid).^0.25, '--', 'Color', [0.85 0.35 0.19], 'LineWidth', 1.5);
plot(V0, n0, 'o', 'Color', [0.85 0.35 0.19], 'MarkerFaceColor', [0.85 0.35 0.19], 'MarkerSize', 8);
xlabel('Membrane voltage V (mV)');
ylabel('K^+ activation gate n');
title('Phase portrait — V vs n');
legend('limit cycle', 'n-nullcline (dn/dt=0)', 'V-nullcline (dV/dt=0)', 'fixed point');
box off;
xlim([-20 120]); ylim([-0.05 0.95]);
saveas(gcf, '../figures/phase_portrait_matlab.png');

fprintf('Done. All three figures saved to figures/\n');

%% ============================================================
%% Local functions
%% ============================================================

function dydt = hh_odes(~, y, I_ext, params)
    V = y(1); m = y(2); h = y(3); n = y(4);
    C_m=params(1); g_Na=params(2); g_K=params(3); g_L=params(4);
    E_Na=params(5); E_K=params(6); E_L=params(7);

    I_Na = g_Na * m^3 * h * (V - E_Na);
    I_K  = g_K  * n^4     * (V - E_K);
    I_L  = g_L            * (V - E_L);

    dV = (I_ext - I_Na - I_K - I_L) / C_m;
    dm = alpha_m(V)*(1-m) - beta_m(V)*m;
    dh = alpha_h(V)*(1-h) - beta_h(V)*h;
    dn = alpha_n(V)*(1-n) - beta_n(V)*n;
    dydt = [dV; dm; dh; dn];
end

function n = count_spikes(V_trace, threshold)
    n = sum(V_trace(1:end-1) < threshold & V_trace(2:end) >= threshold);
end

function a = alpha_m(V); a = 0.1*(25-V) ./ (exp((25-V)/10) - 1); end
function b = beta_m(V);  b = 4.0*exp(-V/18); end
function a = alpha_h(V); a = 0.07*exp(-V/20); end
function b = beta_h(V);  b = 1.0 ./ (exp((30-V)/10) + 1); end
function a = alpha_n(V); a = 0.01*(10-V) ./ (exp((10-V)/10) - 1); end
function b = beta_n(V);  b = 0.125*exp(-V/80); end