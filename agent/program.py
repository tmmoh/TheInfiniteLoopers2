# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from .board import Board
from math import floor, sqrt, log
from random import choice
from time import time
from copy import deepcopy


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


    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.

        remainingTime = referee["time_remaining"]
        print(referee)
        print(remainingTime)
        r = self._root
        print([r.action, r.color, r.visits, r.wins, r.draws])
        match self._color:
            case PlayerColor.RED:
                return self.MCTS(remainingTime)
            case PlayerColor.BLUE:
                return self.MCTS(remainingTime)
                return self.minimax()


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
            case GrowAction():
                print(f"Testing: {color} played GROW action")
            case _:
                raise ValueError(f"Unknown action type: {action}")
        
        self._board.playAction(color, action)
        self._root = self._root.children.get(action, None)
        if self._root == None:
            self._root = MCTS_Node(self._board.currentPlayer.opponent)
        
        

    def minimax(self) -> Action:
        depth = min(1 + floor(self._board.roundNumber ** (1/4)), 
                    Board.MOVE_LIMIT - self._board.roundNumber)
        best_score = Board.MIN_EVAL
        best_move = None

        moveList = self._board.getMoves()
        print(*moveList)
        for move in moveList: 
            '''j
            with open("log.txt", mode="a", encoding="utf-8") as fp:
                fp.write(f'Round {self._board.roundNumber}: Searching through \
                         move {move}\n')
                fp.write(f'{self._color} to play\n')
                '''
            '''
            print("Searching through move", move)
            print(self._color, "to play")
            print(self._board.currentPlayer, "on the board")
            '''
            self.indent += 1
            self._board.playAction(self._board.currentPlayer, move)
            score = self.minimax_value(depth)
            self.indent -= 1
            '''j
            with open("log.txt", mode="a", encoding="utf-8") as fp:
                fp.write(f'Overall score of {score} for move {move}\n')
                '''
            #print("Overall Score of", score, move)
            if(score > best_score):
                best_score = score
                best_move = move
            self._board.undoAction()
            
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
        if self._board.winner != None:
            return self._board.staticEval(self._color)
        
        if depth <= 0:
            return self._board.staticEval(self._color)

        # keep going with minimax

        moveList = self._board.getMoves()

        self.indent -= 1
        '''
        with open("log.txt", mode="a", encoding="utf-8") as fp:
            fp.write('    ' * self.indent + f'{self._board.currentPlayer} \
                    selecting best move\n')
            '''
        self.indent += 1
        # print(self._board.currentPlayer, "selecting best move")
        # Maximising player
        if self._board.currentPlayer == self._color:
            maxEval = Board.MIN_EVAL
            for move in moveList:
                '''
                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    fp.write('    ' * self.indent + f'Trying {move}\n')
                    '''

                self._board.playAction(self._board.currentPlayer, move)
                eval = self.minimax_value(depth - 1, alpha, beta)

                '''
                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    self.indent += 1
                    fp.write('    ' * self.indent + f'{move} with \
                            score {eval}\n')
                    # print(eval, move)
                    '''

                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)

                '''
                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    self.indent -= 1
                    fp.write('    ' * self.indent + f'Undoing {move}\n')
                    '''

                self._board.undoAction()
                if (beta <= alpha):
                    # Prune
                    break
            return maxEval

        else: #Minimising Player - Opponent Move
            minEval = Board.MAX_EVAL
            for move in moveList:
                '''
                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    fp.write(f'Trying {move}\n')
                    '''

                self._board.playAction(self._board.currentPlayer, move)
                eval = self.minimax_value(depth - 1, alpha, beta)

                '''
                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    self.indent += 1
                    fp.write('    ' * self.indent + f'{move} with \
                            score {eval}\n')
                    # print(eval, move)
                    '''

                minEval = min(minEval, eval)
                beta = min(beta, eval)

                '''
                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    self.indent -= 1
                    fp.write('    ' * self.indent + f'Undoing {move}\n')
                    '''

                self._board.undoAction()
                if (beta <= alpha):
                    # Prune
                    break
            return minEval


    def MCTS(self, remaingTime: int) -> Action:
        end = time() + remaingTime / (Board.MOVE_LIMIT - self._board.roundNumber)
        end = time() + 4.5
        print(f'Starting MCTS from board state below')
        print(self._board.render(use_color=True))
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
        #return max(self._root.children.items(), key=lambda item: item[1].visits)[0]
        return max(self._root.children.items(), key=lambda item: item[1].wins / item[1].visits)[0]
   


    def MCTS_Select(self) -> MCTS_Node:
        node = self._root
        while node.fully_expanded(self._board.getMoves()):
            node = node.best_child()
            self._board.playAction(self._board.currentPlayer, node.action)
        
        print(f'Selected leaf node {vars(node)}')
        return node
    
    def MCTS_Expand(self, node: MCTS_Node) -> MCTS_Node:
        
        legal_actions = self._board.getMoves()
        untried = [move for move in legal_actions if move not in node.children]

        # Choose randomly
        action = choice(untried)
        child = MCTS_Node(node.color.opponent, parent=node, action=action)
        node.children[action] = child
        self._board.playAction(self._board.currentPlayer, child.action)

        print(f'Expanded child node {vars(child)}')
        return child or node

    
    def MCTS_Simulate(self, board: Board, child: MCTS_Node) -> PlayerColor | None:
        print(f'Simulating from board state below')
        print(board.render(use_color=True))
        while not board.gameOver():
            moves = board.getMoves()
            move = choice(moves)
            board.playAction(board.currentPlayer, move)
        
        print(f'The Winner was {board.winner}')
        return board.winner

    def MCTS_Backpropogate(self, node: MCTS_Node, result: PlayerColor | None):
        while node != None:
            node.visits += 1

            match result:
                case None:
                    node.draws += 1
                case node.color:
                    node.wins += 1
                case node.color.opponent:
                    pass
            
            node = node.parent