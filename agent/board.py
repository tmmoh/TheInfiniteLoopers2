from dataclasses import dataclass, field
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.board import CellState, BoardMutation, CellMutation
from collections import deque
from math import inf
from copy import deepcopy

BOARD_N = 8
class Board:
    @dataclass(frozen=True, slots=True, order=True)
    class StaticEval:
        pointsAdvantage: int | float
        distance: int | float
        distanceAdvantage: int | float
        history: tuple[int] = tuple()

    MIN_EVAL = StaticEval(-inf, -inf, -inf)
    MAX_EVAL = StaticEval(inf, inf, inf)

    ALL_DIRECTIONS: list[Direction] = [
        Direction.Left, 
        Direction.Right, 
        Direction.DownLeft, 
        Direction.Down, 
        Direction.DownRight,
        Direction.UpLeft,
        Direction.Up,
        Direction.UpRight,
    ]
    RED_MOVES: list[Direction] = [
        Direction.Down, 
        Direction.DownRight,
        Direction.DownLeft, 
        Direction.Right, 
        Direction.Left, 
    ]
    BLUE_MOVES: list[Direction] = [
        Direction.Up,
        Direction.UpLeft,
        Direction.UpRight,
        Direction.Left,
        Direction.Right,
    ]

    MOVE_LIMIT = 150

    roundNumber: int
    currentPlayer: PlayerColor
    _board: dict[Coord, CellState]
    _redFrogs: set[Coord]
    _blueFrogs: set[Coord]
    _history: deque[BoardMutation]
    _redPointHistory: list[int]
    _bluePointHistory: list[int]
    _redStaticEval: StaticEval
    _blueStaticEval: StaticEval
    winner: PlayerColor | None

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
        self._redPointHistory: list[int] = []
        self._bluePointHistory: list[int] = []
        self.winner = None
    
    def render(self, use_color: bool=False, use_unicode: bool=False) -> str:
        """
        Returns a visualisation of the game board as a multiline string, with
        optional ANSI color codes and Unicode characters (if applicable).
        """
        def apply_ansi(str, bold=True, color=None):
            bold_code = "\033[1m" if bold else ""
            color_code = ""
            if color == "RED":
                color_code = "\033[31m"
            if color == "BLUE":
                color_code = "\033[34m"
            if color == "LilyPad":
                color_code = "\033[32m"
            return f"{bold_code}{color_code}{str}\033[0m"

        output = ""
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if self._cell_occupied(Coord(r, c)):
                    state = self._state[Coord(r, c)].state
                    if state == "LilyPad":
                        text = "*"
                    elif state == PlayerColor.RED or state == PlayerColor.BLUE:
                        text = "R" if state == PlayerColor.RED else "B"
                    else:
                        text = " "

                    if use_color:
                        output += apply_ansi(text, color=str(state), bold=False)
                    else:
                        output += text
                else:
                    output += "."
                output += " "
            output += "\n"
        return output

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

        redDist = 0
        redCount = 0
        for frog in self._redFrogs:
            verticalDist = abs(frog.r - Board.winRow(PlayerColor.RED))
            if verticalDist > 0:
                horizontalDist = min([
                    abs(frog.c - cell.c)
                    for i in range(0, BOARD_N)
                    if self._board[
                        cell := Coord(Board.winRow(PlayerColor.RED), i)
                        ] != PlayerColor.RED
                ])
                redDist += max(verticalDist, horizontalDist)
            else:
                redCount += 1
            
        blueDist = 0
        blueCount = 0
        for frog in self._blueFrogs:
            verticalDist = abs(frog.r - Board.winRow(PlayerColor.BLUE))
            if verticalDist > 0:
                horizontalDist = min([
                    abs(frog.c - cell.c)
                    for i in range(0, BOARD_N)
                    if self._board[
                        cell := Coord(Board.winRow(PlayerColor.BLUE), i)
                        ] != PlayerColor.BLUE
                ])
                blueDist += max(verticalDist, horizontalDist)
            else:
                blueCount += 1

        self._redPointHistory.append(-redDist)
        self._bluePointHistory.append(-blueDist)

        # Check for winner
        if redDist == 0 and blueDist > 0:
            self._redStaticEval = Board.StaticEval(inf, blueDist, blueDist)
            self._blueStaticEval = Board.StaticEval(-inf, -blueDist, -blueDist)
            self.winner = PlayerColor.RED
        
        elif blueDist == 0 and redDist > 0:
            self._blueStaticEval = Board.StaticEval(inf, redDist, redDist)
            self._redStaticEval = Board.StaticEval(-inf, -redDist, -redDist)
            self.winner = PlayerColor.BLUE
        
        else:
            self._redStaticEval = Board.StaticEval(redCount - blueCount,
                                                   -redDist,
                                                   blueDist - redDist,
                                                   tuple(self._redPointHistory))
            self._blueStaticEval = Board.StaticEval(blueCount - redCount,
                                                    -blueDist,
                                                    redDist - blueDist,
                                                    tuple(self._bluePointHistory))
            self.winner = None
        


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
        self._redPointHistory.pop()
        self._bluePointHistory.pop()
        

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
    
    def staticEval(self, color: PlayerColor) -> StaticEval:
        match color:
            case PlayerColor.RED:
                return self._redStaticEval
            case PlayerColor.BLUE:
                return self._blueStaticEval

        
    @staticmethod
    def legalMoves(color: PlayerColor) -> list[Direction]:
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

        mutation: BoardMutation = self._growAction(self.currentPlayer)
        if len(mutation.cell_mutations) > 0:
            moves.append(GrowAction())

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
        
    def gameOver(self) -> bool:
        return self.winner != None or self.roundNumber > self.MOVE_LIMIT



