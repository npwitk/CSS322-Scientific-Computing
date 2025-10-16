function result = stable_exp(x, num_terms)
    % STABLE_EXP Compute e^x using stable Taylor series method
    %
    % Use the stable algorithm discussed in class
    % - If x >= 0: compute e^x directly using Taylor series
    % - If x < 0:  compute e^|x| using Taylor series, then return 1/e^|x|
    %
    % Inputs:
    %   x - the input value
    %   num_terms - (optional) number of terms to use in Taylor series
    %               default is 200
    %
    % Output:
    %   result - stable approximation of e^x
    
    if nargin < 2
        num_terms = 200;
    end
    
    if x >= 0
        % For non-negative x, use Taylor series directly
        result = taylor_exp_helper(x, num_terms);
    else
        % For negative x, use stable method:
        % Compute e^|x| and then take reciprocal
        exp_abs_x = taylor_exp_helper(abs(x), num_terms);
        result = 1 / exp_abs_x;
    end
end

function result = taylor_exp_helper(x, num_terms)
    % Helper function to compute e^x using Taylor series for x >= 0
    % Same implementation as the original taylor_exp function
    
    % Initialize result with first term (i=0): x^0/0! = 1
    result = 1;
    
    % Initialize term for iterative computation
    term = 1;
    
    % Compute remaining terms iteratively
    for i = 1:num_terms
        term = term * x / i;  % x^i / i! = (x^(i-1) / (i-1)!) * x / i
        result = result + term;
    end
end

% Also include the original unstable function for comparison
function result = taylor_exp(x, num_terms)
    % Original unstable implementation from Part (a)
    if nargin < 2
        num_terms = 200;
    end
    
    result = 1;
    term = 1;
    
    for i = 1:num_terms
        term = term * x / i;
        result = result + term;
    end
end

%% Comprehensive Test and Comparison Script
fprintf('=== Stable e^x Implementation vs Unstable Version ===\n\n');

% Test values
x_pos = 30;
x_neg = -30;

%% Test for e^30 (positive case)
fprintf('Computing e^30:\n');

% Compute using different methods
stable_pos = stable_exp(x_pos, 200);
unstable_pos = taylor_exp(x_pos, 200);
builtin_pos = exp(x_pos);

fprintf('MATLAB exp(30):           %.15e\n', builtin_pos);
fprintf('Stable method result:     %.15e\n', stable_pos);
fprintf('Unstable method result:   %.15e\n', unstable_pos);
fprintf('\n');

fprintf('Absolute Errors:\n');
stable_abs_err_pos = abs(stable_pos - builtin_pos);
unstable_abs_err_pos = abs(unstable_pos - builtin_pos);
fprintf('Stable method:            %.15e\n', stable_abs_err_pos);
fprintf('Unstable method:          %.15e\n', unstable_abs_err_pos);
fprintf('\n');

fprintf('Relative Errors:\n');
stable_rel_err_pos = stable_abs_err_pos / builtin_pos;
unstable_rel_err_pos = unstable_abs_err_pos / builtin_pos;
fprintf('Stable method:            %.15e\n', stable_rel_err_pos);
fprintf('Unstable method:          %.15e\n', unstable_rel_err_pos);
fprintf('\n');

%% Test for e^(-30) (negative case - where stability matters!)
fprintf('Computing e^(-30):\n');

% Compute using different methods
stable_neg = stable_exp(x_neg, 200);
unstable_neg = taylor_exp(x_neg, 200);
builtin_neg = exp(x_neg);

fprintf('MATLAB exp(-30):          %.15e\n', builtin_neg);
fprintf('Stable method result:     %.15e\n', stable_neg);
fprintf('Unstable method result:   %.15e\n', unstable_neg);
fprintf('\n');

fprintf('Absolute Errors:\n');
stable_abs_err_neg = abs(stable_neg - builtin_neg);
unstable_abs_err_neg = abs(unstable_neg - builtin_neg);
fprintf('Stable method:            %.15e\n', stable_abs_err_neg);
fprintf('Unstable method:          %.15e\n', unstable_abs_err_neg);
fprintf('\n');

fprintf('Relative Errors:\n');
if builtin_neg > 0  % Avoid division by zero
    stable_rel_err_neg = stable_abs_err_neg / builtin_neg;
    unstable_rel_err_neg = unstable_abs_err_neg / builtin_neg;
    fprintf('Stable method:            %.15e\n', stable_rel_err_neg);
    fprintf('Unstable method:          %.15e\n', unstable_rel_err_neg);
else
    fprintf('Cannot compute relative error (true value too small)\n');
end
fprintf('\n');

%% Summary and Analysis
fprintf('=== ANALYSIS AND CONCLUSIONS ===\n');

fprintf('For e^30 (positive x):\n');
if stable_rel_err_pos <= unstable_rel_err_pos
    fprintf('✓ Both methods have similar accuracy (as expected)\n');
    fprintf('  Relative error ratio: %.2f\n', unstable_rel_err_pos / stable_rel_err_pos);
else
    fprintf('• Both methods work well for positive x\n');
end
fprintf('\n');

fprintf('For e^(-30) (negative x):\n');
improvement_factor = unstable_rel_err_neg / stable_rel_err_neg;
fprintf('✓ Stable method is %.1e times more accurate!\n', improvement_factor);
fprintf('  This demonstrates the importance of avoiding catastrophic cancellation\n');
fprintf('\n');

fprintf('Why the stable method works better for negative x:\n');
fprintf('• Unstable method: Computes 1 - 30 + 450 - 4500 + ... (large cancellation)\n');
fprintf('• Stable method:   Computes 1 + 30 + 450 + 4500 + ... then takes 1/result\n');
fprintf('• No catastrophic cancellation in the stable method!\n');
fprintf('\n');
