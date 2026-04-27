function act_lin_idx = get_active_indices(Nv, Nh, Nact, layout)
%GET_ACTIVE_INDICES Return linear indices of active RIS elements for a given layout.
%
%   act_lin_idx = GET_ACTIVE_INDICES(Nv, Nh, Nact, layout)
%
%   Inputs:
%     Nv     : number of vertical elements
%     Nh     : number of horizontal elements
%     Nact   : number of active elements (must be a perfect square, = side_act^2)
%     layout : string specifying placement strategy. One of:
%                'center'   - compact square block at array center (original)
%                'corners'  - one sub-block near each corner
%                'cross'    - elements along center row and column
%                'edges'    - elements at mid-points of the four edges
%                'diagonal' - elements along the main diagonal
%                'random'   - reproducible random placement (fixed seed=42)
%
%   Output:
%     act_lin_idx : Nact-by-1 vector of linear indices into an [Nv x Nh] grid.
%                  Linear indexing follows MATLAB column-major order via sub2ind.
%
%   NOTES:
%     - All layouts try to place exactly Nact elements, adapting geometrically
%       where possible.
%     - 'corners': splits Nact into 4 equal corner sub-blocks; requires Nact
%       divisible by 4 and sqrt(Nact/4) integer. Falls back to 'center' if not.
%     - 'cross': requires Nact even; places Nact/2 elements in the center row
%       and Nact/2 in the center column (intersection counted once).
%     - Calling with the same arguments always returns the same indices, so
%       any function that uses them (generate_channel, ls_dc_estimator, etc.)
%       stays consistent.
%
%   Example (Nv=4, Nh=4, Nact=4):
%     center   -> rows 2-3, cols 2-3
%     corners  -> (1,1),(1,4),(4,1),(4,4)
%     cross    -> (2,1),(2,4),(1,2),(4,2)  [symmetric spread]
%     edges    -> midpoints of each edge
%     diagonal -> (1,1),(2,2),(3,3),(4,4)
%     random   -> 4 random positions with rng seed 42

if nargin < 4 || isempty(layout)
    layout = 'center';
end

layout = lower(layout);

side_act = sqrt(Nact);
if abs(side_act - round(side_act)) > 1e-12
    error('get_active_indices: Nact must be a perfect square.');
end
side_act = round(side_act);

switch layout

    %% ----------------------------------------------------------------
    case 'center'
    %  Original layout: compact square sub-block in the middle.
    %-----------------------------------------------------------------
        v_start   = floor((Nv - side_act)/2) + 1;
        h_start   = floor((Nh - side_act)/2) + 1;
        v_idx_act = v_start:(v_start + side_act - 1);
        h_idx_act = h_start:(h_start + side_act - 1);
        [HhAct, HvAct] = meshgrid(h_idx_act, v_idx_act);
        act_lin_idx = sub2ind([Nv, Nh], HvAct(:), HhAct(:));

    %% ----------------------------------------------------------------
    case 'corners'
    %  Four equal sub-blocks, one near each corner.
    %  Works when Nact is divisible by 4 and sqrt(Nact/4) is integer.
    %  Falls back to 'center' otherwise.
    %-----------------------------------------------------------------
        nCorner = Nact / 4;
        sCorner = sqrt(nCorner);
        if mod(Nact,4)~=0 || abs(sCorner-round(sCorner))>1e-12
            warning(['get_active_indices: ''corners'' requires Nact divisible ' ...
                     'by 4 with integer sqrt(Nact/4). Falling back to ''center''.']);
            act_lin_idx = get_active_indices(Nv, Nh, Nact, 'center');
            return;
        end
        sCorner = round(sCorner);

        % Offsets from each corner (1-based, inward)
        % Top-left, top-right, bottom-left, bottom-right
        v_starts = [1,              1,              Nv-sCorner+1,  Nv-sCorner+1];
        h_starts = [1,              Nh-sCorner+1,   1,             Nh-sCorner+1];

        act_lin_idx = [];
        for c = 1:4
            v_r = v_starts(c):(v_starts(c)+sCorner-1);
            h_r = h_starts(c):(h_starts(c)+sCorner-1);
            [HH, VV] = meshgrid(h_r, v_r);
            act_lin_idx = [act_lin_idx; sub2ind([Nv,Nh], VV(:), HH(:))]; %#ok
        end
        act_lin_idx = unique(act_lin_idx);  % remove any overlap

        % Safety: trim or pad to exactly Nact
        if numel(act_lin_idx) > Nact
            act_lin_idx = act_lin_idx(1:Nact);
        elseif numel(act_lin_idx) < Nact
            warning('get_active_indices: corners layout yielded fewer than Nact; padding.');
            all_idx = setdiff((1:Nv*Nh)', act_lin_idx);
            act_lin_idx = [act_lin_idx; all_idx(1:Nact-numel(act_lin_idx))];
        end

    %% ----------------------------------------------------------------
    case 'cross'
    %  Elements spread along the center row and center column.
    %  Places ceil(Nact/2) on the center row and floor(Nact/2) on the
    %  center column, evenly spaced, with no duplicate at the intersection.
    %-----------------------------------------------------------------
        row_c = round(Nv/2);   % center row
        col_c = round(Nh/2);   % center col

        nRow = ceil(Nact/2);
        nCol = Nact - nRow;   % rest go to column (may include center element)

        % Evenly spaced along center row
        h_idx = unique(round(linspace(1, Nh, nRow)));
        v_row = row_c * ones(size(h_idx));

        % Evenly spaced along center column (avoid intersection with row)
        v_idx_col = unique(round(linspace(1, Nv, nCol+1)));
        v_idx_col(v_idx_col == row_c) = [];   % remove intersection
        v_idx_col = v_idx_col(1:min(nCol, end));
        h_col = col_c * ones(size(v_idx_col));

        v_all = [v_row(:); v_idx_col(:)];
        h_all = [h_idx(:); h_col(:)];
        act_lin_idx = unique(sub2ind([Nv,Nh], v_all, h_all));

        % Trim / pad to exactly Nact
        if numel(act_lin_idx) > Nact
            act_lin_idx = act_lin_idx(1:Nact);
        elseif numel(act_lin_idx) < Nact
            all_idx = setdiff((1:Nv*Nh)', act_lin_idx);
            act_lin_idx = [act_lin_idx; all_idx(1:Nact-numel(act_lin_idx))];
        end

    %% ----------------------------------------------------------------
    case 'edges'
    %  Nact elements placed at (or near) the midpoints of the four edges,
    %  evenly distributed along each edge.
    %  Nact/4 elements per edge; requires Nact divisible by 4.
    %-----------------------------------------------------------------
        if mod(Nact,4) ~= 0
            warning(['get_active_indices: ''edges'' works best when Nact is ' ...
                     'divisible by 4. Using as many as possible.']);
        end
        nPerEdge = max(1, floor(Nact/4));

        % Spread evenly along each edge
        h_pos = unique(round(linspace(1, Nh, nPerEdge)));
        v_pos = unique(round(linspace(1, Nv, nPerEdge)));

        % Top edge (row 1)
        top_v = ones(1, numel(h_pos));   top_h = h_pos;
        % Bottom edge (row Nv)
        bot_v = Nv*ones(1, numel(h_pos)); bot_h = h_pos;
        % Left edge (col 1), exclude corners
        lft_h = ones(1, numel(v_pos));   lft_v = v_pos;
        % Right edge (col Nh), exclude corners
        rgt_h = Nh*ones(1, numel(v_pos)); rgt_v = v_pos;

        v_all = [top_v(:); bot_v(:); lft_v(:); rgt_v(:)];
        h_all = [top_h(:); bot_h(:); lft_h(:); rgt_h(:)];
        act_lin_idx = unique(sub2ind([Nv,Nh], v_all, h_all));

        % Trim / pad
        if numel(act_lin_idx) > Nact
            act_lin_idx = act_lin_idx(1:Nact);
        elseif numel(act_lin_idx) < Nact
            all_idx = setdiff((1:Nv*Nh)', act_lin_idx);
            act_lin_idx = [act_lin_idx; all_idx(1:Nact-numel(act_lin_idx))];
        end

    %% ----------------------------------------------------------------
    case 'diagonal'
    %  Nact elements along the main diagonal (top-left to bottom-right).
    %  Maximises spatial spread in both dimensions simultaneously.
    %-----------------------------------------------------------------
        diag_len = min(Nv, Nh);
        d_idx    = unique(round(linspace(1, diag_len, Nact)));
        % Pad if rounding produced fewer than Nact unique values
        if numel(d_idx) < Nact
            d_idx = [d_idx, setdiff(1:diag_len, d_idx)];
            d_idx = d_idx(1:Nact);
        end
        d_idx = sort(d_idx);
        act_lin_idx = sub2ind([Nv,Nh], d_idx(:), d_idx(:));

        % Trim / pad (in case of duplicates on non-square arrays)
        act_lin_idx = unique(act_lin_idx);
        if numel(act_lin_idx) > Nact
            act_lin_idx = act_lin_idx(1:Nact);
        elseif numel(act_lin_idx) < Nact
            all_idx = setdiff((1:Nv*Nh)', act_lin_idx);
            act_lin_idx = [act_lin_idx; all_idx(1:Nact-numel(act_lin_idx))];
        end

    %% ----------------------------------------------------------------
    case 'random'
    %  Reproducible random placement. Seed is fixed to 42 so that every
    %  call with the same (Nv,Nh,Nact) returns the same layout, ensuring
    %  consistency between generate_channel, ls_dc_estimator, etc.
    %-----------------------------------------------------------------
        rng_state = rng();           % save caller's state
        rng(42, 'twister');          % fixed seed
        perm = randperm(Nv*Nh, Nact);
        rng(rng_state);              % restore caller's state
        act_lin_idx = sort(perm(:));

    %% ----------------------------------------------------------------
    otherwise
        error('get_active_indices: Unknown layout ''%s''.', layout);
end

act_lin_idx = sort(act_lin_idx(:));  % always return sorted column vector

end
