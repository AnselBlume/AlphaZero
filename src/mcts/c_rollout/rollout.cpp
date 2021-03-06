/*

    Simple demo of THC Chess library

    This is a simple "hello world" exercise to get started with the THC chess library
    Just compile and link with thc.cpp. You only need thc.cpp and thc.h to use the
    THC library (see README.MD for more information)

 */

#include <assert.h>
#include <stdio.h>
#include <string>
#include <random>
#include <vector>
#include "thc.h"

// Function declarations
// extern "C" necessary for g++ to export in C compatible format
extern "C" bool isAutomaticDraw(thc::ChessRules&);
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

    return terminal != thc::NOT_TERMINAL || isAutomaticDraw(cr);
}

bool isAutomaticDraw(thc::ChessRules & cr) {
    // Check for 75 move no progress rule
    if (cr.half_move_clock >= 150) {
        return true;
    }

    // Check for fivefold repetition rule
    if (cr.GetRepetitionCount() >= 5) {
        return true;
    }

    // Check for insufficient material
    // First arg doesn't matter as only care about forced draw
    thc::DRAWTYPE result;
    cr.IsInsufficientDraw(false, result);

    return result == thc::DRAWTYPE_INSUFFICIENT_AUTO;
}

int rollout(char* fen) {
    // Random number generator
    static std::random_device rd;
    static std::mt19937 mt(rd());

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

        cr.PlayMove(moves[randInd]);
        moves.clear();
    }
}

int main()
{
    thc::ChessRules cr;
    thc::Move move;
    char * fen;
    int value;

    // White checkmates black and black turn
    fen = "k7/1Q6/1K6/8/8/8/8/8 b - - 0 1";
    value = rollout(fen);
    printf("Value should be -1: %d\n", value);
    assert(value == -1);

    // White checkmates black in one turn and check value
    fen = "k7/2Q5/1K6/8/8/8/8/8 w - - 0 1";

    cr.Forsyth(fen);
    assert(!isTerminalAndValue(cr, true, value));
    assert(value == 0);

    move.TerseIn(&cr, "c7b7");
    cr.PlayMove(move);
    assert(isTerminalAndValue(cr, true, value));
    assert(value == 1);
    printf("Value should be 1: %d\n", value);

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

    // Check draw condition counts increasing
    cr.Forsyth("rnbqkbnr/pp1ppppp/2p5/8/8/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
    for (int i = 0; i < 5; i++) {
        move.TerseIn(&cr, "d1e2");
        cr.PlayMove(move);
        printf("Repetition count: %d\n", cr.GetRepetitionCount());
        printf("Halfmove clock: %d\n", cr.half_move_clock);

        move.TerseIn(&cr, "d8c7");
        cr.PlayMove(move);
        printf("Repetition count: %d\n", cr.GetRepetitionCount());
        printf("Halfmove clock: %d\n", cr.half_move_clock);

        move.TerseIn(&cr, "e2d1");
        cr.PlayMove(move);
        printf("Repetition count: %d\n", cr.GetRepetitionCount());
        printf("Halfmove clock: %d\n", cr.half_move_clock);

        move.TerseIn(&cr, "c7d8");
        cr.PlayMove(move);
        printf("Repetition count: %d\n", cr.GetRepetitionCount());
        printf("Halfmove clock: %d\n", cr.half_move_clock);
    }
}
