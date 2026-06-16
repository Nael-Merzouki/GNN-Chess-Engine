import chess
import torch
from torch_geometric.data import Data
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

    # Create attack edges
    for square, src in piece_nodes.items():
        #piece = board.piece_at(square)

        attacks = board.attacks(square)

        for target in attacks:

            target_piece = board.piece_at(target)

            if target_piece is None:
                continue

            if target not in piece_nodes:
                continue

            if piece is None:
                continue
            
            dst = piece_nodes[target]

            edge_index.append([src, dst])

            if target_piece.color == piece.color:
                edge_type.append(1) # defense edge
            else:
                edge_type.append(0) # attack edge
    
    # Create global edges
    for node in range(global_node):
        edge_index.append([node, global_node])
        edge_type.append(2)
        edge_index.append([global_node, node])
        edge_type.append(2)
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type
    )

    # Store labels
    if value is not None:
        value = math.tanh(value / 400)
        
        data.y_value = torch.tensor(
            [value], dtype=torch.float
        )

    data.policy = policy

    return data