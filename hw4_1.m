D1 = [-1 -1 -1 -1 -1 -1 -1 -1 -1 -11 -1 -1 -1 -1 -1 -1 -1 -1 -1 -11; 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5 -1.5];
D2 = [1 1 1 1 1 1 1 1 1 11 1 1 1 1 1 1 1 1 1 11; 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5 -0.5];
[mu1, mu2, S1, S2, Savg] = a(D1, D2);
b(mu1, mu2, S1, S2, Savg);
function [mu1, mu2, S1, S2, Savg] = a(D1, D2)
    mu1 = (1/length(D1)).*sum(D1,2)

    mu2 = (1/length(D2)).*sum(D2,2)

    sum1 = 0;
    for i = 1:length(D1)
        sum1 = sum1 + (D1(:,i)-mu1)*(D1(:,i)-mu1)';
    end
    S1 = (1/length(D1)).*sum1

    sum2 = 0;
    for i = 1:length(D2)
        sum2 = sum2 + (D2(:,i)-mu2)*(D2(:,i)-mu2)';
    end
    S2 = (1/length(D2)).*sum2

    p1 = length(D1)/(length(D1) + length(D2));
    p2 = length(D2)/(length(D1) + length(D2));
    Savg = p1*S1 + p2*S2
end

function [wLDA, bLDA, CCR] = b(mu1, mu2, S1, S2, Savg)
    wLDA = inv(Savg) * (mu2 - mu1)
end