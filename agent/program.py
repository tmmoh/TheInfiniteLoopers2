# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction, CellState, \
    Board
from copy import deepcopy
import math

BOARD_N = 8
MOVE_LIMIT = 150

    

class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    _ALL_DIRECTIONS: set[Direction] = set([
        Direction.Left, 
        Direction.Right, 
        Direction.DownLeft, 
        Direction.Down, 
        Direction.DownRight,
        Direction.UpLeft,
        Direction.Up,
        Direction.UpRight
    ])
    _RED_MOVES: set[Direction | GrowAction] = set([
        GrowAction, 
        Direction.Left, 
        Direction.Right, 
        Direction.DownLeft, 
        Direction.Down, 
        Direction.DownRight
    ])
    _BLUE_MOVES: set[Direction | GrowAction] = set([
        GrowAction,
        Direction.Left,
        Direction.Right,
        Direction.UpLeft,
        Direction.Up,
        Direction.UpRight
    ])

    _color: PlayerColor
    _opponent: PlayerColor
    _roundNumber: int
    _player: PlayerColor
    _legalMoves: set[Direction | GrowAction]
    _board: dict[Coord, CellState]
    _redFrogs: set[Coord]
    _blueFrogs: set[Coord]

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self._opponent = self._opponent(color)
        self._board = Board()._state
        self._player = PlayerColor.RED
        self._roundNumber = 1

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
                self._legalMoves = self._RED_MOVES
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")
                self._legalMoves = self._BLUE_MOVES
        
        for c in range(1, BOARD_N - 2):
            self._redFrogs.add(Coord(0, c))
            self._blueFrogs.add(Coord(BOARD_N - 1, c))


    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.
        match self._color:
            case PlayerColor.RED:
                print("Testing: RED is playing a MOVE action")
                return MoveAction(
                    Coord(0, 3),
                    [Direction.Down]
                )
            case PlayerColor.BLUE:
                print("Testing: BLUE is playing a GROW action")
                return GrowAction()

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        # There are two possible action types: MOVE and GROW. Below we check
        # which type of action was played and print out the details of the
        # action for demonstration purposes. You should replace this with your
        # own logic to update your agent's internal game state representation.
        match action:
            case MoveAction(coord, dirs):
                dirs_text = ", ".join([str(dir) for dir in dirs])
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Directions: {dirs_text}")

                startCoord = coord
                endCoord = startCoord

                for dir in dirs:
                    endCoord += dir
                    if self._frogCell(coord):
                        endCoord += dir

                self._board[endCoord] = self._board.pop(startCoord)
                self._frogs(color).remove(startCoord) 
                self._frogs(color).add(endCoord) 

                '''
                # Keep track of all lilypad cells

                # Whenever a frog moves to a new cell,
                # check if surrounding cells can convert for each color
                self._redPads.remove(endCoord)
                self._bluePads.remove(endCoord)

                for dir in self._ALL_DIRECTIONS:
                    nextCoord = endCoord + dir
                    if self._inBounds(nextCoord) and nextCoord not in self._board:
                        self._pads(color).add(nextCoord)

                # Whenever a frog leaves a cell,
                # check if current and surrounding cells can convert for each color
                '''
                

            case GrowAction():
                print(f"Testing: {color} played GROW action")
            case _:
                raise ValueError(f"Unknown action type: {action}")
        
        self._roundNumber += 1
        self._player = self._opponent(self._player)
    

    def _legalMoves(self, color: PlayerColor) -> set[Direction | GrowAction]:
        match color:
            case PlayerColor.RED:
                return self._RED_MOVES
            case PlayerColor.BLUE:
                return self._BLUE_MOVES

    def _frogs(self, color: PlayerColor) -> set[Coord]:
        match color:
            case PlayerColor.RED:
                return self._redFrogs
            case PlayerColor.BLUE:
                return self._blueFrogs
    
    def _pads(self, color: PlayerColor) -> set[Coord]:
        match color:
            case PlayerColor.RED:
                return self._redPads
            case PlayerColor.BLUE:
                return self._bluePads
    
    def _opponent(color: PlayerColor) -> PlayerColor:
        match color:
            case PlayerColor.RED:
                return PlayerColor.BLUE
            case PlayerColor.BLUE:
                return PlayerColor.RED

    def _frogCell(self, coord: Coord) -> bool:
        return (self._board.get(coord) == CellState(PlayerColor.RED)
                or self._board.get(coord) == CellState(PlayerColor.BLUE))

    def _growAction(self, color: PlayerColor) -> bool: 
        for frog in self._frogs(color):
            for dir in self._ALL_DIRECTIONS:
                nextCoord = frog + dir
                if self._inBounds(nextCoord) and nextCoord not in self._board:
                    self._board[nextCoord] = CellState("LilyPad")
        
    def _inBounds(coord: Coord) -> bool:
        return (coord.r >= 0 and coord.r < BOARD_N 
            and coord.c >= 0 and coord.c < BOARD_N)

    def _winRow(color: PlayerColor) -> int:
        match color:
            case PlayerColor.RED:
                return BOARD_N - 1
            case PlayerColor.BLUE:
                return 0

            
        
        

    def minimax() -> MoveAction:
        move_couter{'count': MOVE_LIMIT - roundNumber}
        best_score = -float('inf')
        best_move = None
        
        for move in moveList: 
            score = minimax_value()
            if(score > best_score):
                best_score = score
                best_move = move
            
        return best_move 
    
    def minimax_value() -> int:
        if('''Terminal Node''')
            '''FUNCTION THAT CHECKS IF ALL FROGS ARE AT FINAL ROW FOR BOTH RED OR BLUE'''
            return #Evaluate state
        
        if('''Move limit reached''')
            return #Evaluate state
        
        if ('''Our move'''): #Maximising Player
            best_score = -float('inf')
            for move in moveList:
                new_state = apply_move
                score = minimax_value('''Opponent to move''')
            return best_score
        else: #Minimising Player - Opponent Move
            best_score = float('inf')
            for move in movelist:
                new_state = apply_move
                score = minimax_value('''Our move''')
            
            return


    def minimax_value(self, depth: int, playerFrogs: set[Coord], opponentFrogs: set[Coord], color: PlayerColor) -> int:

        playerDist = 0
        for frog in playerFrogs:
            playerDist += abs(frog.r - self._winRow(self._color))
            
        opponentDist = 0
        for frog in opponentFrogs:
            opponentDist += abs(frog.r - self._winRow(self._opponent(self._color)))

        # Check for winner
        if playerDist == 0 and opponentDist != 0:
            return math.inf
        
        if opponentDist == 0 and playerDist != 0:
            return -math.inf
        
        if depth <= 0:
            return opponentDist - playerDist

        # keep going with minimax

        # Maximising player
        if color == self._color:
            maxEval = -math.inf
            for move in moveList:
                new_state = apply_move
                eval = minimax_value('''Opponent to move''')
                maxEval = max(maxEval, eval)
            return bestScore

        else: #Minimising Player - Opponent Move
            best_score = math.inf
            for move in movelist:
                new_state = apply_move
                eval = minimax_value('''Our move''')
                minEval = min(minEval, eval)
            return best_score


            

            
    MOVE_COST = 1
    GOAL_ROW = BOARD_N - 1
    RED_DIRECTIONS: list[Direction] = [Direction.DownRight, Direction.Left, Direction.Right, Direction.DownLeft, Direction.Down]

    def getMoves(board: dict[Coord, CellState], redFrogCoords: Coord) -> list[MoveAction]:
        moves: list[MoveAction] = []

        for dir in RED_DIRECTIONS:
            # Check that we're in bounds and not wrapping around
            if redFrogCoords.r + dir.r not in range(BOARD_N):
                continue
            if redFrogCoords.c + dir.c not in range(BOARD_N):
                continue

            nextCoord: Coord = redFrogCoords + dir
            if board.get(nextCoord, None) == CellState.LILY_PAD:
                #print(redFrogCoords, nextCoord, board.get(nextCoord, None))
                # Single, immediate jump
                moves.append(MoveAction(redFrogCoords, dir))

        # Calculate possible other multi-jumps
        getJumpMoves(board, moves, redFrogCoords, redFrogCoords, MoveAction(redFrogCoords, []))

        return moves


    # Recursive function
    def getJumpMoves(board: dict[Coord, CellState], moves: list[MoveAction], currCoord: Coord, prevCoord: Coord, move: MoveAction):
        for dir in RED_DIRECTIONS:
            # Check that we're in bounds and not wrapping around
            if currCoord.r + 2 * dir.r not in range(BOARD_N):
                continue
            if currCoord.c + 2 * dir.c not in range(BOARD_N):
                continue

            nextCoord = currCoord + dir
            nextCell = board.get(nextCoord, None)

            jumpCoord = currCoord + dir * 2
            jumpCell = board.get(jumpCoord, None)
            if jumpCell == CellState.LILY_PAD and (nextCell == CellState.BLUE or nextCell == CellState.RED):
                if jumpCoord == prevCoord:
                    # Don't jump back to where you already were
                    # Prevents infinite loop
                    continue

                moveCopy = deepcopy(move)
                moveCopy.directions.append(dir)
                moves.append(moveCopy)
                
                # Copy and recurse
                getJumpMoves(board, moves, jumpCoord, currCoord, moveCopy)
