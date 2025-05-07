# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.board import CellState, BoardMutation, CellMutation
from copy import deepcopy
from collections import deque
import math

from referee.game.player import Player


BOARD_N = 8
MOVE_LIMIT = 150


class Board:
    ALL_DIRECTIONS: set[Direction] = set([
        Direction.Left, 
        Direction.Right, 
        Direction.DownLeft, 
        Direction.Down, 
        Direction.DownRight,
        Direction.UpLeft,
        Direction.Up,
        Direction.UpRight
    ])
    RED_MOVES: set[Direction | GrowAction] = set([
        GrowAction(), 
        Direction.Left, 
        Direction.Right, 
        Direction.DownLeft, 
        Direction.Down, 
        Direction.DownRight
    ])
    BLUE_MOVES: set[Direction | GrowAction] = set([
        GrowAction(),
        Direction.Left,
        Direction.Right,
        Direction.UpLeft,
        Direction.Up,
        Direction.UpRight
    ])

    roundNumber: int
    currentPlayer: PlayerColor
    _board: dict[Coord, CellState]
    _redFrogs: set[Coord]
    _blueFrogs: set[Coord]
    _history: deque

    def __init__(self):
        self._board: dict[Coord, CellState] = {
            Coord(r, c): CellState() 
            for r in range(BOARD_N) 
            for c in range(BOARD_N)
        }

        for r in [0, BOARD_N - 1]:
            for c in [0, BOARD_N - 1]:
                self._board[Coord(r, c)] = CellState("LilyPad")

        for r in [1, BOARD_N - 2]:
            for c in range(1, BOARD_N - 1):
                self._board[Coord(r, c)] = CellState("LilyPad")
            
        for c in range(1, BOARD_N - 1):
            self._board[Coord(0, c)] = CellState(PlayerColor.RED)
            self._redFrogs.add(Coord(0, c))
            self._board[Coord(BOARD_N - 1, c)] = CellState(PlayerColor.BLUE)
            self._blueFrogs.add(Coord(BOARD_N - 1, c))

        self.currentPlayer: PlayerColor = PlayerColor.RED
        self.roundNumber: int = 1
        self._history: deque[BoardMutation] = deque()

    def playAction(self, color: PlayerColor, action: Action):
        mutation: BoardMutation
        match action:
            case MoveAction(coord, dirs):
                dirs_text = ", ".join([str(dir) for dir in dirs])
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Directions: {dirs_text}")
                mutation = self._moveAction(color, action)
            case GrowAction():
                print(f"Testing: {color} played GROW action")
                mutation = self._growAction(color)
            case _:
                raise ValueError(f"Unknown action type: {action}")

        for mut in mutation.cell_mutations:
            match self._board[mut.cell].state:
                case PlayerColor.RED:
                    self._redFrogs.remove(mut.cell)
                case PlayerColor.BLUE:
                    self._blueFrogs.remove(mut.cell)

            self._board[mut.cell] = mut.next

            match self._board[mut.cell].state:
                case PlayerColor.RED:
                    self._redFrogs.add(mut.cell)
                case PlayerColor.BLUE:
                    self._blueFrogs.add(mut.cell)

        
        self.currentPlayer = self.currentPlayer.opponent
        self.roundNumber += 1
        self._history.append(mutation)

    def undoAction(self):
        mutation: BoardMutation = self._history.pop()

        for mut in mutation.cell_mutations:
            match self._board[mut.cell].state:
                case PlayerColor.RED:
                    self._redFrogs.remove(mut.cell)
                case PlayerColor.BLUE:
                    self._blueFrogs.remove(mut.cell)

            self._board[mut.cell] = mut.prev

            match self._board[mut.cell].state:
                case PlayerColor.RED:
                    self._redFrogs.add(mut.cell)
                case PlayerColor.BLUE:
                    self._blueFrogs.add(mut.cell)

        self.currentPlayer = self.currentPlayer.opponent
        self.roundNumber -= 1
        

    def _frogs(self, color: PlayerColor) -> set[Coord]:
        match color:
            case PlayerColor.RED:
                return self._redFrogs
            case PlayerColor.BLUE:
                return self._blueFrogs
    
    def _isFrogCell(self, coord: Coord) -> bool:
        return self._board[coord].state == PlayerColor

    def _isEmptyCell(self, coord: Coord) -> bool:
        return self._board[coord].state == None

    def _moveAction(self, color: PlayerColor, action: MoveAction) -> BoardMutation:
        startCoord = action.coord
        endCoord = startCoord

        for dir in action.directions:
            endCoord += dir
            if self._isFrogCell(endCoord):
                endCoord += dir

        '''
        self._board[endCoord] = self._board.pop(startCoord)
        self._frogs(color).remove(startCoord) 
        self._frogs(color).add(endCoord) 
        '''

        cellMutations = {
            startCoord: CellMutation(
                startCoord,
                self._board[startCoord],
                CellState(None)
            ),
            endCoord: CellMutation(
                endCoord,
                self._board[endCoord],
                self._board[startCoord]
            )
        }

        return BoardMutation(
            action,
            cell_mutations=set(cellMutations.values())
        )



    def _growAction(self, color: PlayerColor) -> BoardMutation: 
        cellMutations = {}
        neighbour_cells: set[Coord] = set()

        for cell in self._frogs(color):
            for direction in Direction:
                try:
                    neighbour = cell + direction
                    neighbour_cells.add(neighbour)
                except ValueError:
                    pass

        for cell in neighbour_cells:
            if self._isEmptyCell(cell):
                cellMutations[cell] = CellMutation(
                    cell,
                    self._board[cell],
                    CellState("LilyPad")
                )

        return BoardMutation(
            GrowAction(),
            cell_mutations=set(cellMutations.values())
        )

        
    @staticmethod
    def legalMoves(color: PlayerColor) -> set[Direction | GrowAction]:
        match color:
            case PlayerColor.RED:
                return Board.RED_MOVES
            case PlayerColor.BLUE:
                return Board.BLUE_MOVES

    @staticmethod
    def inBounds(coord: Coord) -> bool:
        return (coord.r >= 0 and coord.r < BOARD_N 
            and coord.c >= 0 and coord.c < BOARD_N)

    @staticmethod
    def winRow(color: PlayerColor) -> int:
        match color:
            case PlayerColor.RED:
                return BOARD_N - 1
            case PlayerColor.BLUE:
                return 0

    

class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    _color: PlayerColor
    _opponent: PlayerColor
    _legalMoves: set[Direction | GrowAction]
    _board: Board

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self._opponent = color.opponent
        self._board = Board()
        self._legalMoves = Board.legalMoves(self._color)

        


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
        
        self._board.playAction(color, action)


            
        
        

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
