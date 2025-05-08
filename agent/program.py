# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from .board import Board
from math import floor, inf


BOARD_N = 8




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
        
        

    def minimax(self) -> Action:
        depth = min(1 + floor(self._board.roundNumber ** (1/4)), 
                    Board.MOVE_LIMIT - self._board.roundNumber)
        best_score = Board.MIN_EVAL
        best_move = None

        moveList = self._board.getMoves()
        print(*moveList)
        for move in moveList: 
            with open("log.txt", mode="a", encoding="utf-8") as fp:
                fp.write(f'Round {self._board.roundNumber}: Searching through \
                         move {move}\n')
                fp.write(f'{self._color} to play\n')
            '''
            print("Searching through move", move)
            print(self._color, "to play")
            print(self._board.currentPlayer, "on the board")
            '''
            self.indent += 1
            self._board.playAction(self._board.currentPlayer, move)
            score = self.minimax_value(depth)
            self.indent -= 1
            with open("log.txt", mode="a", encoding="utf-8") as fp:
                fp.write(f'Overall score of {score} for move {move}\n')
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
        with open("log.txt", mode="a", encoding="utf-8") as fp:
            fp.write('    ' * self.indent + f'{self._board.currentPlayer} \
                    selecting best move\n')
        self.indent += 1
        # print(self._board.currentPlayer, "selecting best move")
        # Maximising player
        if self._board.currentPlayer == self._color:
            maxEval = Board.MIN_EVAL
            for move in moveList:
                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    fp.write('    ' * self.indent + f'Trying {move}\n')

                self._board.playAction(self._board.currentPlayer, move)
                eval = self.minimax_value(depth - 1, alpha, beta)

                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    self.indent += 1
                    fp.write('    ' * self.indent + f'{move} with \
                            score {eval}\n')
                    # print(eval, move)

                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)

                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    self.indent -= 1
                    fp.write('    ' * self.indent + f'Undoing {move}\n')

                self._board.undoAction()
                if (beta <= alpha):
                    # Prune
                    break
            return maxEval

        else: #Minimising Player - Opponent Move
            minEval = Board.MAX_EVAL
            for move in moveList:
                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    fp.write(f'Trying {move}\n')

                self._board.playAction(self._board.currentPlayer, move)
                eval = self.minimax_value(depth - 1, alpha, beta)

                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    self.indent += 1
                    fp.write('    ' * self.indent + f'{move} with \
                            score {eval}\n')
                    # print(eval, move)

                minEval = min(minEval, eval)
                beta = min(beta, eval)

                with open("log.txt", mode="a", encoding="utf-8") as fp:
                    self.indent -= 1
                    fp.write('    ' * self.indent + f'Undoing {move}\n')

                self._board.undoAction()
                if (beta <= alpha):
                    # Prune
                    break
            return minEval




'''
class MCTS_node:
    def __init__(self, parent=None, action: Action = None):
        self.parent = parent
        self.action = action
        self.children: list[MCTS_node] = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = board.getMoves() #How to store board ??

    def fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=1):
        choices = [
            (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices.index(max(choices))]
    
    def expand(self):
        action = self.untried_actions.pop()
        #play the action
        child_node = MCTS_node(parent=self, action=action)
        self.children.append(child_node)
        return child_node
    
    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(-result)


class Agent_MCTS:

    def MCTS(self) -> Action:
        root = MCTS_node()

        while(''Iteration time left''):
            node = root

            #Selection
            while node.fully_expanded() and node.children:
                node = best_child()

            #Expansion 
            if not node.fully_expanded():
                #Expand the node
                node = node.expand()

            #Simulation
            result = self.simulation()

            #Backpropagation
            backpropagate(result)
        
    return

    def simulation():
        while(''Simulation Limit''):
            #Make random move
            
            #Check win condition
            #If we win, return 1
            #If opponent win, return -1
            return

        #If simulation time exceeded return no color (Draw/unfinished)
    return None   
    
    '''

    

            


