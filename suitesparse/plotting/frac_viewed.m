cd C:\Users\nhajl\Desktop\Current_Work\greedy_sparse_clean\suitesparse\plotting\CDF_plots\

%%% Combine current results
addpath("../../unsparsified")
combine_results()

%%% Load in the results 
load("greedy_unsparsified.mat"); % M_greedy
load("lanczos_unsparsified.mat"); % M_lanczos
load("power_unsparsified.mat"); % M_power

%%% Load in basic matrix info
% load("../../matrix_info/matinfo_results_1_to_1131"); % M

%%% PARAMETERS
init_cond = 'diag'; % which initial condition to plot
relerr = 5; % which relative errors to plot (10^(-relerr))
plot_folder = ''; % which folder to send plots
other = 'lanczos'; % can be lanczos or can be power

%%% MATLAB PLOTTING PARAMETERS
set(gca, 'Position',  [100, 100, 2000, 800]);
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');




% Getting and Filtering out data
if strcmpi(other, 'lanczos')
    M_other = M_lanczos;
else
    M_other = M_power;
end

greedy_viewed = M_greedy.(sprintf('greedy_antistallA_%s', init_cond));
other_viewed = M_other.(sprintf('%s_antistallA_%s', other, init_cond));

% get list of filenames of matrices that don't get NaN for requested relerr
greedy_keep = M_greedy.filenames(find(~isnan(greedy_viewed(relerr, :))));
other_keep = M_other.filenames(find(~isnan(other_viewed(relerr, :))));
filenames_keep = intersect(greedy_keep, other_keep);

% remove diagonal matrices, smaller matrices
% matrices_big_enough = M.filenames(find(M.ns > 100));
% matrices_nondiag = M.filenames(find(~M.is_diag));
% matrices_keep = intersect(matrices_nondiag, matrices_big_enough);
% filenames_keep = intersect(filenames_keep, matrices_keep);
length(filenames_keep)

% get indices corresponding to filenames for both methods
[~, greedy_inds] = ismember(filenames_keep, M_greedy.filenames);
[~, other_inds] = ismember(filenames_keep, M_other.filenames);


%%% PLOTTING
r = relerr;
clf;
plotname = sprintf('Fraction of Entries Read to Reach $10^{-%d}$ Relative Error', r);
plotfilename = sprintf("%s%s_%s_viewed_relerr_neg_%d.png", plot_folder, init_cond, other, r);
legend_names = ["greedy", other];

% Plot the CDF for unsparsified greedy at relerr r
[sorted_greedy, yvals_greedy] = get_yvals(greedy_viewed(r, greedy_inds));
semilogx(sorted_greedy, yvals_greedy, 'LineWidth', 3);
hold on; 

% Plot the CDF for unsparsified other method at relerr r
[sorted_other, yvals_other] = get_yvals(other_viewed(r, other_inds));
semilogx(sorted_other, yvals_other, 'LineWidth', 3);
hold on; 

% --- Add vertical & horizontal lines for Greedy at x = 1e-3 ---
x_target = 1e-3;
% find nearest index on greedy curve
[~, idx] = min(abs(sorted_greedy - x_target));
x_val = sorted_greedy(idx);
y_val = yvals_greedy(idx);

% vertical line
plot([x_val x_val], [0 y_val], 'r', 'LineWidth', 2);
hold on;
% horizontal line
plot([10^(-5.5) x_val], [y_val y_val], 'r', 'LineWidth', 2);
hold on;


%vertInd = 0:.1:1;
legend(legend_names);
xlabel('fraction of entries read','FontSize',20,'interpreter','latex');
ylabel('$\%$ of matrices with $\leq x$ of entries read','FontSize',20,'interpreter','latex');
xlim([10^(-5.5), 1])
legend('location', 'Northwest', 'FontSize', 15)
title(plotname,'FontSize',20,'interpreter','latex')
exportgraphics(gca,plotfilename,'Resolution',600) 
hold off


function [sorted, yvals] = get_yvals(A)
    sorted = sort(A);
    yvals = zeros(1,length(sorted));
    for i = 1:length(sorted)
       yvals(i) = sum(sorted <= sorted(i));
    end
    yvals = yvals/length(sorted);
end