%% Tempo Ratios
tempo_origin = 120;

ratio = [
    2,1; % 8ve
    3,2; % P5
    4,3; % P4
    5,4;
    6,5;
    1024,729;
    36,25;
    256,243; % m2
    ];
tempo = zeros(length(ratio),2);
for i=1:length(ratio)
    tempo(i,1) = tempo_origin;
    tempo(i,2) = (tempo_origin/ratio(i,1))*ratio(i,2);
end
tempo