function [Ht, dH] = constant_h(v,theta_h)

assert(theta_h >= 0);
assert(theta_h <= 1);

Ht = ones(size(v))*theta_h;

dH = ones(size(v));
