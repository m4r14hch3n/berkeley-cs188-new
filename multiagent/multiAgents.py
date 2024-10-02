# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState
from layout import Layout

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        print("legalMoves:", legalMoves)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print("scores:", scores)
        bestScore = max(scores)
        print("bestScore:", bestScore)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        print("chosenIndex:", chosenIndex)
        print("legalMoves[chosenIndex]:", legalMoves[chosenIndex])
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        print("newPos: ",newPos)
        newFood = successorGameState.getFood()
        print("newFood: ",newFood)

        newGhostStates = successorGameState.getGhostStates()

        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # return successorGameState.getScore()

        score = successorGameState.getScore()
        print("original score:", score)

        distance = float("inf")
        for single_newFoodPos in newFood.asList():
            # print("single_newFoodPos:", single_newFoodPos)
            distance_temp = util.manhattanDistance(newPos, single_newFoodPos)
            if distance_temp < distance:
                distance = distance_temp
        score += 10.0 / distance
        print("score after considering new food:", score)
        
        for ghostState in newGhostStates:
            scaredTime = ghostState.scaredTimer
            ghostPos = ghostState.getPosition()
            ghostDistance = util.manhattanDistance(newPos, ghostPos)
            print("ghostDistance:", ghostDistance)
            print("scaredTime:", scaredTime)

            if scaredTime > 0:
                score += 20.0 / ghostDistance
            else:
                if ghostDistance <= 2:
                    score -= 10
                else:
                    score -= 10.0 / ghostDistance
        print("score after considering ghost:", score)
        
        return score

# used for adversarial search agents
def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:
                maxEval = float("-inf")
                pacmanLegalActions = gameState.getLegalActions(agentIndex)

                for action in pacmanLegalActions:
                    newGameState = gameState.generateSuccessor(agentIndex, action)
                    eval = minimax(1, depth, newGameState)
                    if eval > maxEval:
                        maxEval = eval
                        bestAction = action
                if depth == 0:
                    return bestAction
                else:
                    return maxEval
            else:
                minEval = float("inf")
                nextIndex = (agentIndex + 1) % (gameState.getNumAgents())
                if nextIndex == 0:
                    depth += 1
                ghostLegalActions = gameState.getLegalActions(agentIndex)
                for action in ghostLegalActions:
                    newGameState = gameState.generateSuccessor(agentIndex, action)
                    eval = minimax(nextIndex, depth, newGameState)
                    minEval = min(minEval, eval)
                return minEval
            
        return minimax(0, 0, gameState)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # alpha: 当前max层可以保证的最大值，初始值-∞； beta：当前min层可以保证的最小值，初始值+∞
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:
                value = float("-inf")
                bestAction = None
                pacmanLegalActions = gameState.getLegalActions(agentIndex)

                for action in pacmanLegalActions:
                    newGameState = gameState.generateSuccessor(agentIndex, action)
                    score = alphabeta(1, depth, newGameState, alpha, beta)
                    if score>value:
                        value=score
                        bestAction = action
                    alpha = max(alpha, value)
                    if alpha > beta:
                        break
                if depth == 0:
                    return bestAction
                else:
                    return value
            else:
                value = float("inf")
                nextAgent = (agentIndex + 1) % (gameState.getNumAgents())
                if nextAgent == 0:
                    depth += 1
                ghostLegalActions = gameState.getLegalActions(agentIndex)
                for action in ghostLegalActions:
                    newGameState = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(nextAgent, depth, newGameState, alpha, beta))
                    beta = min(value, beta)
                    if alpha > beta:
                        break
                return value
            
        return alphabeta(0, 0, gameState, float("-inf"), float("inf"))
      


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
                if gameState.isWin() or gameState.isLose() or depth == self.depth:
                    return self.evaluationFunction(gameState)
                
                if agentIndex == 0:
                    maxEval = float("-inf")
                    pacmanLegalActions = gameState.getLegalActions(agentIndex)

                    for action in pacmanLegalActions:
                        newGameState = gameState.generateSuccessor(agentIndex, action)
                        eval = expectimax(1, depth, newGameState)
                        if eval > maxEval:
                            maxEval = eval
                            bestAction = action
                    if depth == 0:
                        return bestAction
                    else:
                        return maxEval
                else:
                    minEval = float("inf")
                    nextIndex = (agentIndex + 1) % (gameState.getNumAgents())
                    if nextIndex == 0:
                        depth += 1
                    ghostLegalActions = gameState.getLegalActions(agentIndex)
                    length = len(ghostLegalActions)
                    total = 0
                    for action in ghostLegalActions:
                        newGameState = gameState.generateSuccessor(agentIndex, action)
                        eval = expectimax(nextIndex, depth, newGameState)
                        total += eval
                    expecti = total/length
                    return expecti
                
        return expectimax(0, 0, gameState)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <1. distance to the food: score += a / distance<pacManPos, foodPos>;
    2. threat from unscrared ghost: score -= b / distance<pacManPos, ghostPos>;
    3. hunt scared ghost: score += c / distance<pacManPos, ghostPos>;
    4. nab pellet: score += d / distance<pacManPos, pelletPos>;
    >
    """
    "*** YOUR CODE HERE ***"
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # return successorGameState.getScore()

    score = currentGameState.getScore()
    # print("original score:", score)

    distance = float("inf")
    for single_newFoodPos in newFood.asList():
        # print("single_newFoodPos:", single_newFoodPos)
        distance_temp = util.manhattanDistance(newPos, single_newFoodPos)
        if distance_temp < distance:
            distance = distance_temp
    score += 10.0 / distance
    # print("score after considering new food:", score)
    
    for ghostState in newGhostStates:
        scaredTime = ghostState.scaredTimer
        ghostPos = ghostState.getPosition()
        ghostDistance = util.manhattanDistance(newPos, ghostPos)
        # print("ghostDistance:", ghostDistance)
        # print("scaredTime:", scaredTime)

        if scaredTime > 0:
            score += 20.0 / ghostDistance
        else:
            if ghostDistance <= 2:
                score -= 10
            else:
                score -= 10.0 / ghostDistance
    # print("score after considering ghost:", score)

    for capsule in capsules:
        # print(capsule)
        capsuleDistance = util.manhattanDistance(newPos, capsule)
        score += 10 / capsuleDistance
    
    return score



# Abbreviation
better = betterEvaluationFunction
