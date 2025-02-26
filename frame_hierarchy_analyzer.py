# frame_hierarchy_analyzer.py
import json
import os
from typing import List, Set
import logging

frame_relations = {
    "Inheritance": ["Inherits from", "Is Inherited by"],
    "Perspective": ["Perspective on", "Is Perspectivized in"],
    "Usage": ["Uses", "Is Used by"],
    "Subframe": ["Subframe of", "Has Subframe(s)"],
}

class FrameNode(object):
    def __init__(self, name: str, encoding: str = "utf-8"):
        self.name = name
        self.next = {}
        assert encoding in ["utf-8", "ascii"]
        self.encoding = encoding
        return
    
    def __str__(self):
        return self._str_helper()
    
    def _sorted_next_list(self):
        return [item[1] for item in sorted(self.next.items())]
    
    def _str_helper(self, is_head=True, prefix="", is_tail=False):
        if is_head:
            result = self.name + '\n'
        else:
            if self.encoding == "utf-8":
                result = prefix + ("└── " if is_tail else "├── ") + self.name + '\n'
            else:
                result = prefix + ("+-- " if is_tail else "+-- ") + self.name + '\n'
        for i, node in enumerate(self._sorted_next_list()):
            if is_head:
                new_prefix = prefix
            else:
                if self.encoding == "utf-8":
                    new_prefix = prefix + ("    " if is_tail else "│   ")
                else:
                    new_prefix = prefix + ("    " if is_tail else "|   ")
            result += node._str_helper(False, new_prefix, i == len(self.next) - 1)
        return result
    
    def find(self, node_name: str):
        if self.name == node_name:
            return self
        for node in self.next.values():
            result = node.find(node_name)
            if result:
                return result
        return None
    
    def delete(self, node_name: str):
        self.next = {key:val for key, val in self.next.items() if key != node_name}
        return
    
    def count_nodes(self):
        return 1 + sum([subnode.count_nodes() for subnode in self.next.values()])
    
    def children(self):
        return list(self.next.values())
    
class RootFrameNode(FrameNode):
    def append_root(self, node: FrameNode, parents: list=[]):
        if not parents:
            self.next[node.name] = node
        else:
            for parent in parents:
                parent_node = FrameNode(name=parent, encoding=self.encoding)
                parent_node.next[node.name] = node
                self.next[parent_node.name] = parent_node
            if node in self.next.values():
                self.delete(node.name)
        return
    
def get_frames(foldername, suffix=".xml"):
    return [file[:-4] for file in os.listdir(foldername) if file[-4:] == suffix]

def get_all_frames() -> List[str]:
    """Get all frames from frame_json directory."""
    return [file[:-5] for file in os.listdir("frame_json") if file.endswith('.json')]

def get_related_frames(frame: str, frame_relation: str, reverse_order: bool, all_frames: List[str]) -> Set[str]:
    """Recursively find all related frames based on the relation and direction."""
    related_frames = set()
    try:
        with open(f"frame_json/{frame}.json", 'r') as fo:
            data_dict = json.load(fo)
        relation_key = frame_relations[frame_relation][0 if not reverse_order else 1]  # e.g., "Inherits from" or "Is Inherited by"
        relations = data_dict.get("fr_rel", {}).get(relation_key, "").split(', ')
        relations = [r.strip() for r in relations if r.strip()]
        
        for related_frame in relations:
            if related_frame in all_frames:
                related_frames.add(related_frame)
                related_frames.update(get_related_frames(related_frame, frame_relation, reverse_order, all_frames))
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # Ignore missing or malformed files for custom frames
    return related_frames

def analyze_hierarchy(
        frames: list, 
        frame_relation: str, 
        reverse_order: bool = False, 
        encoding: str = "utf-8"
    ):
    """Analyzes and builds the frame hierarchy based on the specified relation, including all related frames recursively."""
    assert frame_relation in frame_relations, f'''Please enter one of the relations: ['{"', '".join(frame_relations.keys())}']'''
    root = RootFrameNode(f"[{frame_relations[frame_relation][1 if not reverse_order else 0]}]", encoding=encoding)
    
    # Get all frames from frame_json to include related frames
    all_frames = get_all_frames()
    combined_frames = set(frames)  # Start with user-provided frames

    # Recursively expand to include all related frames
    for frame in list(combined_frames):
        related = get_related_frames(frame, frame_relation, reverse_order, all_frames)
        combined_frames.update(related)

    for frame in combined_frames:
        is_node_existing = False
        
        try:
            with open(f"frame_json/{frame}.json", 'r') as fo:
                data_dict = json.load(fo)
            frame_name = data_dict.get("frame_name", frame).strip()
            if frame_name in root.next.keys():
                node = root.next[frame_name]
                is_node_existing = True
            else:
                node = FrameNode(frame_name, encoding=encoding)
            parents = data_dict.get("fr_rel", {}).get(frame_relations[frame_relation][0 if not reverse_order else 1], "")
            parents = [parent.strip() for parent in parents.split(', ')] if parents else []

            if not parents and not is_node_existing:
                root.append_root(node)
            elif parents:
                for parent in parents:
                    parent_node = root.find(parent)
                    if parent_node:
                        parent_node.next[node.name] = node
                        if node in root.next.values():
                            root.delete(node.name)
                    else:
                        root.append_root(node, [parent])
        except FileNotFoundError:
            node = FrameNode(frame, encoding=encoding)
            root.append_root(node)
        except Exception as e:
            logging.error(f"Error processing frame {frame}: {str(e)}")
            continue
    return root

def save_hierarchy_to_file(root, filename):
    with open(filename, 'w') as fo:
        fo.write(str(root))
    print(f"Hierarchy has been saved to {filename}!")