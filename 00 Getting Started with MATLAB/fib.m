function v = fib(n)
    if n < 0
        error('Invalid input');
    elseif n <= 1
        v = n;
    else
        v = fib(n-1) + fib(n-2);
    end
end