from dataclasses import dataclass
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.board import CellState, BoardMutation, CellMutation
from collections import deque
from math import inf

BOARD_N = 8
class Board:
    ALL_DIRECTIONS: set[Direction] = set([
        Direction.Left, 
        Direction.Right, 
        Direction.DownLeft, 
        Direction.Down, 
        Direction.DownRight,
        Direction.UpLeft,
        Direction.Up,
        Direction.UpRight,
    ])
    RED_MOVES: set[Direction] = set([
        Direction.Down, 
        Direction.DownLeft, 
        Direction.DownRight,
        Direction.Left, 
        Direction.Right, 
    ])
    BLUE_MOVES: set[Direction] = set([
        Direction.Up,
        Direction.UpLeft,
        Direction.UpRight,
        Direction.Left,
        Direction.Right,
    ])

    MOVE_LIMIT = 150

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

        self._redFrogs = set()
        self._blueFrogs = set()

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
                mutation = self._moveAction(color, action)
            case GrowAction():
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
        return (self._board[coord].state == PlayerColor.BLUE 
            or self._board[coord].state == PlayerColor.RED)

    def _isPadCell(self, coord: Coord) -> bool:
        return self._board[coord].state == "LilyPad"

    def _isEmptyCell(self, coord: Coord) -> bool:
        return self._board[coord].state == None

    def _moveAction(
            self, 
            color: PlayerColor, 
            action: MoveAction
            ) -> BoardMutation:
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
    def legalMoves(color: PlayerColor) -> set[Direction]:
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

    def getMoves(self) -> list[Action]:
        moves: list[Action] = []
        mutation: BoardMutation = self._growAction(self.currentPlayer)
        if len(mutation.cell_mutations) > 0:
            moves.append(GrowAction())

        for frog in self._frogs(self.currentPlayer):
            # Calculate jumps and multi-jumps for each frog
            # First because they're likely better moves, helps with pruning
            self.getJumpMoves(moves, frog)

        for move in self.legalMoves(self.currentPlayer):
            # Order by directions then by frogs because direction is more
            # valuable, should help with pruning
            for frog in self._frogs(self.currentPlayer):
                try:
                    nextCoord: Coord = frog + move
                    if self._isPadCell(nextCoord):
                        moves.append(MoveAction(frog, move))
                except ValueError:
                    pass


        return moves


    # Recursive function
    def getJumpMoves(
            self, 
            moves: list[Action], 
            currCoord: Coord, prevCoord: Coord | None = None, 
            prevMove: MoveAction | None = None
            ):
        for dir in self.legalMoves(self.currentPlayer):
            try:
                nextCoord: Coord = currCoord + dir
                jumpCoord = currCoord + dir * 2

                if self._isPadCell(jumpCoord) and self._isFrogCell(nextCoord):
                    if jumpCoord == prevCoord:
                        # Don't jump back to where you already were
                        # Prevents infinite loop
                        # There are no other ways to create cycles
                        continue
                    
                    moveAction: MoveAction
                    if not prevMove:
                        moveAction = MoveAction(currCoord, (dir,))
                    else:
                        moveAction = MoveAction(
                                prevMove.coord, 
                                prevMove.directions + (dir,)
                                )

                    moves.append(moveAction)
                    
                    # Copy and recurse
                    self.getJumpMoves(
                            moves, 
                            jumpCoord, 
                            prevCoord=currCoord, 
                            prevMove=moveAction
                            )
            except ValueError:
                pass

    @dataclass(frozen=True, slots=True, order=True)
    class StaticEval:
        pointsAdvantage: int | float
        distance: int | float
        distanceAdvantage: int | float

    MIN_EVAL = StaticEval(-inf, -inf, -inf)
    MAX_EVAL = StaticEval(inf, inf, inf)


