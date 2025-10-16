function result = taylor_exp(x, num_terms)
    % TAYLOR_EXP Compute e^x using Taylor series expansion
    %
    % Inputs:
    %   x - the input value
    %   num_terms - number of terms to use in Taylor series; default is 200
    %
    % Output:
    %   result - approximation of e^x using Taylor series
    %
    % Formula: e^x = sum_{i=0}^{n} (x^i / i!)
    
    if nargin < 2
        num_terms = 200; % If the user calls taylor_exp(x) with only one argument
    end
    
    result = 1;
    term = 1;
    
    % Compute remaining terms iteratively
    for i = 1:num_terms
        term = term * x / i;  % x^i / i! = (x^(i-1) / (i-1)!) * x / i
        result = result + term;
    end
end

%% Test and Analysis Script
fprintf('=== Taylor Series e^x Implementation Test ===\n\n');

% Test values
x_pos = 30;
x_neg = -30;

% Compute using Taylor series
taylor_pos = taylor_exp(x_pos, 200);
taylor_neg = taylor_exp(x_neg, 200);

builtin_pos = exp(x_pos);
builtin_neg = exp(x_neg);

% Display results
fprintf('Computing e^30:\n');
fprintf('Taylor series result: %.15e\n', taylor_pos);
fprintf('Built-in exp result:  %.15e\n', builtin_pos);
fprintf('Absolute error:       %.15e\n', abs(taylor_pos - builtin_pos));
fprintf('Relative error:       %.15e\n', abs(taylor_pos - builtin_pos) / builtin_pos);
fprintf('\n');

fprintf('Computing e^(-30):\n');
fprintf('Taylor series result: %.15e\n', taylor_neg);
fprintf('Built-in exp result:  %.15e\n', builtin_neg);
fprintf('Absolute error:       %.15e\n', abs(taylor_neg - builtin_neg));
if builtin_neg ~= 0
    fprintf('Relative error:       %.15e\n', abs(taylor_neg - builtin_neg) / builtin_neg);
else
    fprintf('Relative error:       Cannot compute (built-in result is 0)\n');
end
fprintf('\n');

%% Analysis of why Taylor series is unstable for negative x

fprintf('=== Stability Analysis ===\n\n');

% Let's look at individual terms for e^(-30)
fprintf('First 20 terms of Taylor series for e^(-30):\n');
x = -30;
term = 1;
fprintf('Term %2d: %15.6e\n', 0, term);

for i = 1:19
    term = term * x / i;
    fprintf('Term %2d: %15.6e\n', i, term);
end

fprintf('\nObservations:\n');
fprintf('1. For e^(-30), many early terms are very large in magnitude\n');
fprintf('2. These large terms alternate in sign (positive/negative)\n');
fprintf('3. The final result should be very small: e^(-30) ≈ %.2e\n', exp(-30));
fprintf('4. This leads to catastrophic cancellation - subtracting large numbers\n');
fprintf('   to get a small result loses precision\n\n');