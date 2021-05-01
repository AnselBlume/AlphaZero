r'''
    Policy represented by 8x8x73
    8x8: which square to pick up a piece from
    0-55: number of squares to move a piece in a dir {1,...,7} and dir to
    move it {N, NE, E, SE, S, SW, W, NW} organized as
    {N1,...,N7, NE1,...,NE7, ... ,W1,...W7,...,NW1,...,NW7}

    56-63: Knight moves |-, __|, --|, |_, _|, |--, |__, -|

    64-72: Pawn underpromotions in each possible movement/capture direction
    \N, \B, \R, |N, |B, |R, /N, /B, /R
'''
# Direct indices into the array
POLICY_N_OFFSET = 0
POLICY_NE_OFFSET = 7
POLICY_E_OFFSET = 14
POLICY_SE_OFFSET = 21
POLICY_S_OFFSET = 28
POLICY_SW_OFFSET = 35
POLICY_W_OFFSET = 42
POLICY_NW_OFFSET = 49

POLICY_KNIGHT_OFFSET = 56
POLICY_KNIGHT_UP_RIGHT = 56
POLICY_KNIGHT_RIGHT_UP = 57
POLICY_KNIGHT_RIGHT_DOWN = 58
POLICY_KNIGHT_DOWN_RIGHT = 59
POLICY_KNIGHT_DOWN_LEFT = 60
POLICY_KNIGHT_LEFT_DOWN = 61
POLICY_KNIGHT_LEFT_UP = 62
POLICY_KNIGHT_UP_LEFT = 63

POLICY_PROMOTION_OFFSET = 64
POLICY_PROMOTION_VW = 64
POLICY_PROMOTION_V = 67
POLICY_PROMOTION_VE = 70

# Offsets within the policy promotion directions (to be added to those)
POLICY_PROMOTION_KNIGHT = 0
POLICY_PROMOTION_BISHOP = 1
POLICY_PROMOTION_ROOK = 2
