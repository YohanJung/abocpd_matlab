function inc(myvar,incval)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INC - increments a variable in the local workspace
%
% Syntax:  INC variable (number)
%
%          Call INC in the command form:
%
%          inc i 10
%
%          will increment the variable 'i' by 10 in the
%          local workspace.  Number is optional.  If left
%          out, the default increment value is 1.
%
% Output:  There is no output.  The variable specified is
%          silently incremented in the local workspace.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%written by Jeffrey Johnson 1-13-04

if nargin < 2
  incval = 1;
end

yy = [myvar ' = ' myvar ' + ' num2str(incval) ';'];

evalin('caller',yy)







