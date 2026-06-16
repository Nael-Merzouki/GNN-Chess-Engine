from ast import If

import chess
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import math


PIECE_MAP = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

def fen_to_graph(fen, value=None, policy=None):
    board = chess.Board(fen)

    node_features = []
    edge_index = []
    edge_type = []

    piece_nodes = {}

    pin_edges = []
    pin_edge_type = []

    ray_edges = []
    ray_edge_type = []

    # Create piece nodes
    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None:
            continue
        
        node_id = len(node_features)
        piece_nodes[square] = node_id

        file = chess.square_file(square)
        rank = chess.square_rank(square)

        mobility = len(board.attacks(square))

        is_pinned = float(board.is_pinned(piece.color, square))
        
        # Attack & Defense 
        defense_count = 0
        attack_count = 0

        for target_square in board.attacks(square):

            target_piece = board.piece_at(target_square)

            if target_piece is None:
                continue

            if target_piece.color == piece.color:
                defense_count += 1
            else:
                attack_count += 1

        node_features.append([
            PIECE_MAP[piece.piece_type] / 6.0,
            1 if piece.color == chess.WHITE else 0,
            file / 7.0,
            rank / 7.0,
            mobility / 27.0,
            is_pinned,
            attack_count / 16.0,
            defense_count / 16.0
        ])
    
    # Create global node
    global_node = len(node_features)
    node_features.append([0.0, 
                          float(board.turn), 
                          float(board.has_kingside_castling_rights(chess.WHITE)), 
                          float(board.has_kingside_castling_rights(chess.BLACK)), 
                          float(board.has_queenside_castling_rights(chess.WHITE)), 
                          float(board.has_queenside_castling_rights(chess.BLACK)), 
                          board.fullmove_number / 100.0, 
                          0.0])

    # Create attack/defense & king-zone edges
    for square, src in piece_nodes.items():
        piece = board.piece_at(square)

        if piece is None:
            continue

        if board.is_pinned(piece.color, square):
            attackers = board.attackers(not piece.color, square)

            for attacker_sq in attackers:
                attacker_id = piece_nodes[attacker_sq]

                pin_edges.append([attacker_id, node_id])
                pin_edge_type.append(7)

        attacks = board.attacks(square)

        for target in attacks:

            target_piece = board.piece_at(target)

            if target_piece is None:
                continue

            if target not in piece_nodes:
                continue
            
            dst = piece_nodes[target]

            edge_index.append([src, dst])

            if target_piece.color == piece.color:
                edge_type.append(1) # defense edge
            else:
                edge_type.append(0) # attack edge
        
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)

        white_king_id = piece_nodes[white_king_square]
        black_king_id = piece_nodes[black_king_square]

        if (piece.color != chess.WHITE) and (square != white_king_square):
            if chess.square_distance(square, white_king_square) <= 2:
                edge_index.append([src, white_king_id])
                edge_type.append(3)
        
        if (piece.color != chess.BLACK) and (square != black_king_square):
            if chess.square_distance(square, black_king_square) <= 2:
                edge_index.append([src, black_king_id])
                edge_type.append(3)

    
    # Create global edges
    for node in range(global_node):
        edge_index.append([node, global_node])
        edge_type.append(4)
        edge_index.append([global_node, node])
        edge_type.append(4)
    
    for sq1, id1 in piece_nodes.items():
        piece1 = board.piece_at(sq1)
        if piece1 is None:
            continue
        for sq2, id2 in piece_nodes.items():
            piece2 = chess.piece_at(sq2)
            if piece2 is None:
                continue

            if sq1 == sq2:
                continue
            
            f1 = chess.square_file(sq1)
            r1 = chess.square_rank(sq1)

            f2 = chess.square_file(sq2)
            r2 = chess.square_rank(sq2)

            # Same-file edges
            #if (f1 == f2) or (r1 == r2):

            #    if can_slide_piece(piece1, 'file') and is_clear_ray(board, sq1, sq2):
            #        ray_edges.append([id1, id2])
            #        ray_edge_type.append()
                #edge_index.append([id1, id2])
                #edge_type.append(3)
                #continue

            #if abs(f1 - f2) == abs(r1 - r2):

            #    if can_slide_piece(piece1, 'diag') and is_clear_ray(board, sq1, sq2):
            #        ray_edges.append([id1, id2])
            #        ray_edge_type.append()
            # Same-rank edges
            #if r1 == r2:
            #    edge_index.append([id1, id2])
            #    edge_type.append(4)
            #    continue

            # Same-diagonal edges
            #if abs(f1 - f2) == abs(r1 - r2):
            #    edge_index.append([id1, id2])
            #    edge_type.append(5)
            #    continue
    


    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # One hot encoding of edge types
    edge_attr = F.one_hot(
        torch.tensor(edge_type),
        num_classes=3
    ).float()

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_attr
    )

    # Store labels
    if value is not None:
        value = math.tanh(value / 400)
        
        data.y_value = torch.tensor(
            [value], dtype=torch.float
        )

    data.policy = policy

    return data

def is_clear_ray(board, sq1, sq2):
    f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
    f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)

    df = f2 - f1
    dr = r2 - r1

    if not (df == 0 or dr == 0 or abs(df) == abs(dr)):
        return False
    
    step_f = 0 if df == 0 else (1 if df > 0 else -1)
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)

    f, r = f1 + step_f, r1 + step_r

    while (f, r) != (f2, r2):
        sq = chess.square(f, r)
        if board.piece_at(sq) is not None:
            return False
        f += step_f
        r += step_r
    
    return True

def can_slide_piece(piece, direction_type):
    if piece.piece_type == chess.QUEEN:
        return True
    
    if direction_type in ('file', 'rank') and piece.piece_type == chess.ROOK:
        return True
    
    if direction_type == 'diag' and piece.piece_type == chess.BISHOP:
        return True
    
    return False