/*

    Simple demo of THC Chess library

    This is a simple "hello world" exercise to get started with the THC chess library
    Just compile and link with thc.cpp. You only need thc.cpp and thc.h to use the
    THC library (see README.MD for more information)

 */

#include <stdio.h>
#include <string>
#include <random>
#include <vector>
#include "thc.h"

// Function declarations
// extern "C" necessary for g++ to export in C compatible format
extern "C" bool isDraw(thc::ChessRules&);
extern "C" bool isTerminalAndValue(thc::ChessRules&, bool, int&);
extern "C" int rollout(char*);
extern "C" void display_position(thc::ChessRules, std::string&);

void display_position( thc::ChessRules &cr, const std::string &description )
{
    std::string fen = cr.ForsythPublish(); // Mangle this name because conflicts with main
    std::string s = cr.ToDebugStr();
    printf( "%s\n", description.c_str() );
    printf( "FEN (Forsyth Edwards Notation) = %s\n", fen.c_str() );
    printf( "Position = %s\n", s.c_str() );
}

bool isTerminalAndValue(thc::ChessRules & cr, bool isOrigWhite, int & value) {
    thc::TERMINAL terminal;
    cr.Evaluate(terminal);

    // Compute the termination value of the state based on the original player
    switch (terminal) {
        case thc::TERMINAL_WCHECKMATE:
            value = isOrigWhite ? -1 : 1;
            break;
        case thc::TERMINAL_BCHECKMATE:
            value = isOrigWhite ? 1 : -1;
            break;
        // Stalemate or not terminal
        default: // This will always set the value to 0 if there isn't a checkmate
            value = 0;
    }

    return terminal != thc::NOT_TERMINAL || isDraw(cr);
}

bool isDraw(thc::ChessRules & cr) {
    thc::DRAWTYPE result;

    // First arg doesn't matter as only care about forced draw
    cr.IsInsufficientDraw(false, result);
    return result == thc::DRAWTYPE_INSUFFICIENT_AUTO;
}

int rollout(char* fen) {
    // Random number generator
    static std::random_device rd;
    static std::mt19937 mt(rd());

    // Assumes the current fen isn't terminal
    thc::ChessRules cr;
    cr.Forsyth(fen);
    bool isOrigWhite = cr.WhiteToPlay();

    std::vector<thc::Move> moves;
    int value;

    while (true) {
        //display_position(cr, "");
        if (isTerminalAndValue(cr, isOrigWhite, value)) {
            return value;
        }
        cr.GenLegalMoveList(moves);

        std::uniform_int_distribution<> rand(0, moves.size() - 1);
        int randInd = rand(mt);

        cr.PushMove(moves[randInd]);
        moves.clear();
    }
}

int main()
{
    // White checkmates black and black turn
    char * fen = "k7/1Q6/1K6/8/8/8/8/8 b - - 0 1";
    int value = rollout(fen);
    printf("Value should be -1: %d\n", value);

    // Draw
    fen = "k7/8/4K3/8/8/8/8/8 b - - 0 1";
    value = rollout(fen);
    printf("Value should be 0: %d\n", value);

    // Stalemate
    fen = "k7/7R/1R6/8/8/1K6/8/8 b - - 0 1";
    value = rollout(fen);
    printf("Value should be 0: %d\n", value);

    // Rollout from start
    printf("Rolling out from start\n");
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    value = rollout(fen);
    printf("Value should be arbitrary: %d\n", value);
}
