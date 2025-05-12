# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

import collections
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from .board import Board, CellState
from math import floor, sqrt, log
from random import choice, choices
from time import time
from copy import deepcopy
from collections import defaultdict


BOARD_N = 8


class MCTS_Node:
    def __init__(self, color: PlayerColor, parent: 'MCTS_Node' = None, action: Action = None):
        self.parent: MCTS_Node = parent
        self.action: Action = action
        self.color: PlayerColor = color
        self.children: dict[Action, MCTS_Node] = {}
        self.visits = 0
        self.wins = 0
        self.draws = 0

    def fully_expanded(self, legal_actions):
        return all(action in self.children for action in legal_actions)
    
    def best_child(self, c_param=sqrt(2)):
        return max(
            self.children.values(),
            key=lambda child: (child.wins / child.visits) + c_param * sqrt(log(self.visits) / child.visits)
        )
        ## INCLUDING DRAWS AS WINS


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    DEPTH_LIMIT = 5

    _color: PlayerColor
    _opponent: PlayerColor
    _legalMoves: list[Direction]
    _board: Board
    _root: MCTS_Node
    _evaluations: dict[tuple[tuple[Coord, CellState], ...], 
                       dict[tuple[int, PlayerColor], 
                            tuple[Action, Board.StaticEval]]]
    _usedTime: float

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self._opponent = color.opponent
        self._board = Board()
        self._legalMoves = Board.legalMoves(self._color)
        self._root = MCTS_Node(self._board.currentPlayer.opponent)
        self._usedTime = 0
        self._evaluations = defaultdict(dict)



    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
            


        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.

        print(referee)
        # fx = 0.0026 * (-x*x + 75*x)
        # allocatedTime = self._usedTime - usedTime
        # allocatedTime =  2 * remainingTime / (1 + Board.MOVE_LIMIT - self._board.roundNumber)
        #allocatedTime =  2 * 180 / 75 * (-x/75 + 1)

        remainingTime: float = referee["time_remaining"]
        actualUsedTime = 180 - remainingTime
        x = (self._board.roundNumber + 1) // 2
        fx = 0.000065 * x * (x-75) * (x-75)
        timeAllocated = max(self._usedTime/actualUsedTime, 0.1) * fx
    
        print(f'{self._color}\'s {x} move\n \
                {timeAllocated} seconds allocated\n \
                {self._usedTime} seconds previously allocated\n \
                {actualUsedTime} seconds actually used\n')

        self._usedTime += fx

        return self.minimax(timeAllocated)
        '''
        print(remainingTime)
        r = self._root
        print([r.action, r.color, r.visits, r.wins, r.draws])
        return self.MCTS(remainingTime)
        '''
        match self._color:
            case PlayerColor.RED:
                if self._board.roundNumber < 25:
                    return self.minimax(timeAllocated)
                return self.MCTS(timeAllocated)
            case PlayerColor.BLUE:
                return self.minimax(timeAllocated)


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
        self._root = self._root.children.get(action, None)
        if self._root == None:
            self._root = MCTS_Node(self._board.currentPlayer.opponent)
        
        

    def minimax(self, allocatedTime: float) -> Action:
        moveList = self._board.getMoves()
        best_score = Board.MIN_EVAL
        best_move = choice(moveList)

        # print(*moveList)
        
        ### Iterative Deepening Approach
        depth = 0
        maxTime = time() + allocatedTime
        timeTaken = 0
        while timeTaken * depth < maxTime - time():
            start = time()
            # Every time we go deeper, we'll either take at least the same
            # amount of time as the depth before
            # If the remaining time isn't enough for that, save resources
            depth += 1
            print(f'Enough time for depth {depth}')
            best_score = Board.MIN_EVAL
            best_move = choice(moveList)

            dct = self._evaluations.get(self._board.state(), None)
            if dct != None:
                eval = dct.get((depth, self._board.currentPlayer), None)
                if eval != None:
                    move, score = eval
                    if score > best_score:
                        best_score = score
                        best_move = move
                    timeTaken = time() - start
                    print(f"Found evaluation for depth {depth} with action {move}")
                    break

            for move in moveList: 
                self._board.playAction(self._board.currentPlayer, move)
                score = self.minimax_value(depth)

                if score > best_score:
                    best_score = score
                    best_move = move

                self._board.undoAction()
                x = self._evaluations[self._board.state()]
                x[(depth, self._board.currentPlayer)] = (move, score)

            timeTaken = time() - start
            
        print("Best Score:", best_score)
        return best_move 
    
    indent = 1
    def minimax_value(
            self, 
            depth: int, 
            alpha: Board.StaticEval = Board.MIN_EVAL, 
            beta: Board.StaticEval = Board.MAX_EVAL
            ) -> Board.StaticEval:


        # Check for winner
        if self._board.gameOver():
            return self._board.staticEval(self._color)
        
        if depth <= 0:
            return self._board.staticEval(self._color)

        moveList = self._board.getMoves()

        # Maximising player

        if self._board.currentPlayer == self._color:
            maxEval = Board.MIN_EVAL
            for move in moveList:
                self._board.playAction(self._board.currentPlayer, move)
                eval = self.minimax_value(depth - 1, alpha, beta)

                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)

                self._board.undoAction()
                if (beta <= alpha):
                    # Prune
                    break
            return maxEval

        else: #Minimising Player - Opponent Move
            minEval = Board.MAX_EVAL
            for move in moveList:
                self._board.playAction(self._board.currentPlayer, move)
                eval = self.minimax_value(depth - 1, alpha, beta)

                minEval = min(minEval, eval)
                beta = min(beta, eval)

                self._board.undoAction()
                if (beta <= alpha):
                    # Prune
                    break
            return minEval


    def MCTS(self, remainingTime: float) -> Action:
        end = time() + 2 * remainingTime / (1 + Board.MOVE_LIMIT - self._board.roundNumber)
        #end = time() + 1.5
        print(f'Starting MCTS from board state below')
        print(self._board.render(use_color=True))
        end = time() + remainingTime
        while time() < end:
            leaf = self.MCTS_Select()
            child = self.MCTS_Expand(leaf)
            result = self.MCTS_Simulate(deepcopy(self._board), child)
            self.MCTS_Backpropogate(child, result)

            # Undo board for selection
            while child != self._root:
                self._board.undoAction()
                child = child.parent

        r = self._root
        print(f'Ended MCTS on board state below')
        print(self._board.render(use_color=True))
        print([r.action, r.color, r.visits, r.wins, r.draws])
        return max(self._root.children.items(), key=lambda item: item[1].visits)[0]
        #return max(self._root.children.items(), key=lambda item: item[1].wins / item[1].visits)[0]
   


    def MCTS_Select(self) -> MCTS_Node:
        node = self._root
        while node.fully_expanded(self._board.getMoves()):
            node = node.best_child()
            self._board.playAction(self._board.currentPlayer, node.action)
        
        #print(f'Selected leaf node {vars(node)}')
        return node
    
    def MCTS_Expand(self, node: MCTS_Node) -> MCTS_Node:
        
        legal_actions = self._board.getMoves()
        untried = [move for move in legal_actions if move not in node.children]

        # Choose randomly
        action = choice(untried)
        child = MCTS_Node(node.color.opponent, parent=node, action=action)
        node.children[action] = child
        self._board.playAction(self._board.currentPlayer, child.action)

        #print(f'Expanded child node {vars(child)}')
        return child or node

    
    def MCTS_Simulate(self, board: Board, child: MCTS_Node) -> PlayerColor | None:
        #print(f'Simulating from board state below')
        #print(board.render(use_color=True))
        while not board.gameOver():
            moves = board.getMoves()
            #move = choice(moves)
            weights = []
            for move in moves:
                weighting = 1
                if type(move) == MoveAction:
                    for dir in move.directions:
                        if dir in [Direction.Up, Direction.UpLeft, Direction.UpRight,
                            Direction.Down, Direction.DownLeft, Direction.DownRight]:
                            weighting += 1

                weights.append(weighting)

            move = choices(population=moves, weights=weights)[0]
            board.playAction(board.currentPlayer, move)
        
        #print(f'The Winner was {board.winner}')
        #print(board.render(use_color=True))
        return board.winner

    def MCTS_Backpropogate(self, node: MCTS_Node, result: PlayerColor | None):
        while node != None:
            r = node
            #print([r.color, r.visits, r.wins, r.draws, r.action])
            node.visits += 1

            match result:
                case None:
                    node.draws += 1
                case node.color:
                    node.wins += 1
                case node.color.opponent:
                    pass

            #print([r.color, r.visits, r.wins, r.draws, r.action])
            node = node.parent
