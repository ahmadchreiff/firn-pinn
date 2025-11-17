function GenerateFirnData()
    % Generate firn forward data on 129x129 grid with the paper parameters

    %-----------------------------
    % 1. Grid
    %-----------------------------
    h  = 1/128;           % mesh size in z and t
    z  = 0:h:1;           % 129 points in [0,1]
    t  = 0:h:1;           % 129 points in [0,1]
    n  = numel(z);
    m  = numel(t);

    %-----------------------------
    % 2. Physical parameters (Table 2)
    %-----------------------------
    zF = 1;
    Te = 1;
    f  = 0.2;

    Ma  = 0.04 * 9.8 / (8.314 * 260);   % ≈ 1.8134e-4
    G   = 10 + 0.03;                    % tau + lambda
    F   = 200 + 485;                    % v + w_air

    % csts vector as in DirectPbResc comments
    % csts(1)=n, csts(2)=m, csts(3)=zF, csts(4)=dt,
    % csts(5)=T, csts(6)=G/f, csts(7)=1/f, csts(8)=Ma/f, csts(9)=F
    csts          = zeros(1,9);
    csts(1) = n;
    csts(2) = m;
    csts(3) = zF;
    csts(4) = h;          % dt = h
    csts(5) = Te;         % T = 1
    csts(6) = G / f;
    csts(7) = 1 / f;
    csts(8) = Ma / f;
    csts(9) = F;

    %-----------------------------
    % 3. Coefficients on the grid
    %-----------------------------
    % D_alpha(z) = 200 - 199.98*z
    Da = 200 - 199.98 * z(:);     % column vector of length n

    % rho_atm(t) = 2 * (Te * t)^(1/4)
    v0 = 2 * (Te * t).^(1/4);     % row vector of length m

    % finite element matrices
    hvec      = diff(z)';         % length n-1
    [M,K]     = COEFFc(hvec, n, F);
    [A,S]     = COEFFv(hvec, n, Da);

    %-----------------------------
    % 4. Solve the direct problem
    %-----------------------------
    V = DirectPbResc(z, Da, v0, A, S, M, K, csts);  % n x m

    %-----------------------------
    % 5. Save to your repo
    %-----------------------------
    % Go one level up (..) to repo root, then into data/raw
    outPath = fullfile('..', 'data', 'raw', 'firn_forward.mat');
    save(outPath, 'z', 't', 'V', 'Da', 'v0', 'csts');

    fprintf('Saved firn_forward.mat to %s\n', outPath);
end
