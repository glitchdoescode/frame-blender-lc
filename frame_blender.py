#frame_blender.py
from frame_hierarchy_analyzer import get_frames, analyze_hierarchy
from evaluation_data_service import EvaluationDataService
from prompts import Prompts
import curses
import threading
import os
import sys

from langchain.prompts import PromptTemplate
from langchain_service import LangChainService

FRAME_FOLDER = "frame"
FRAME_JSON_FOLDER = "frame_json"
EVALUATION_CONFIG = {
    "data_directory": "data",
    "data_file": "evaluation.json"
}


### Window Classes (Unchanged) ###

class Window:
    def __init__(
            self,
            title: str = "",
            begin_y: int = 0,
            begin_x: int = 0,
            nlines: int = 0,
            ncols: int = 0,
            content: str = "",
            center: bool = False
    ):
        self.title = title
        self.begin_y = begin_y
        self.begin_x = begin_x
        self.center = center
        self.focus = False
        self.start_line = 0

        if ncols:
            self.ncols = ncols
        else:
            max_line_len = max([len(i) for i in content.split('\n')])
            title_len = len(title)
            max_len = stdscr_width - begin_x
            self.ncols = min(max_len, max(max_line_len, title_len))
        self.content = self.split_lines(content, self.ncols)
        if nlines:
            self.nlines = nlines
        else:
            self.nlines = self.count_lines()

        self.win = curses.newwin(self.nlines + 2, self.ncols + 2, begin_y, begin_x)
        self.update_content(self.content)
        return

    def end_yx(self):
        return self.nlines + 2 + self.begin_y, self.ncols + 2 + self.begin_x

    def update_focus(self, focus: bool):
        if focus:
            self.focus = True
            self.win.addstr(0, max(0, (self.ncols + 2 - len(self.title)) // 2), self.title, curses.color_pair(1) | curses.A_BOLD | curses.A_UNDERLINE)
        else:
            self.focus = False
            self.win.addstr(0, max(0, (self.ncols + 2 - len(self.title)) // 2), self.title)
        self.win.refresh()
        return

    def split_lines(self, content, maxcols):
        if not content:
            return ""
        lines_tmp = content.replace('\t', '    ').split('\n')
        lines = []
        while lines_tmp:
            if not lines_tmp[0]:
                lines_tmp.pop(0)
                continue
            lines.append(lines_tmp[0][:maxcols])
            lines_tmp[0] = lines_tmp[0][maxcols:]
        return '\n'.join(lines)

    def count_lines(self):
        return len(self.content.split('\n'))

    def scroll_down(self):
        self.start_line = min(self.start_line + 1, self.count_lines() - self.nlines)
        self.update_content()
        return

    def scroll_up(self):
        self.start_line = max(self.start_line - 1, 0)
        self.update_content()
        return

    def update_content(self, content: str = None, attr: int = 0):
        if content is None:
            content = self.content
        else:
            self.content = content
        self.win.clear()
        self.win.border()
        self.update_focus(self.focus)
        lines = content.replace('\t', '    ').split('\n')[self.start_line: self.start_line + self.nlines]
        start_cols = []
        for i, line in enumerate(lines):
            if self.center:
                start_col = (self.ncols - min(len(line), self.ncols)) // 2 + 1
            else:
                start_col = 1
            self.win.addstr(i + 1, start_col, line[:self.ncols], attr)
            start_cols.append(start_col - 1)
        self.win.refresh()
        return lines, start_cols

class Input:
    def __init__(self):
        self.cursor_x = 0
        self.content = ""
        return

    def move_cursor_right(self):
        self.cursor_x = min(self.cursor_x + 1, len(self.content))

    def move_cursor_left(self):
        self.cursor_x = max(self.cursor_x - 1, 0)

    def delete_char(self):
        if self.cursor_x > 0:
            self.content = self.content[:self.cursor_x - 1] + self.content[self.cursor_x:]
            self.move_cursor_left()

    def add_char(self, char):
        self.content = self.content[:self.cursor_x] + char + self.content[self.cursor_x:]
        self.move_cursor_right()

    def execute(self, key):
        if key == curses.KEY_BACKSPACE or key == 127:
            self.delete_char()
        elif key == curses.KEY_LEFT:
            self.move_cursor_left()
        elif key == curses.KEY_RIGHT:
            self.move_cursor_right()
        else:
            self.add_char(chr(key))

class InputWindow(Window):
    def __init__(self, *args, prompt='', **kwargs):
        self.prompt = prompt
        self.confirmed = False
        self.input = Input()
        super().__init__(*args, content=prompt, **kwargs)
        self.id = "input"
        return

    def update_cursor(self):
        if self.focus:
            curses.curs_set(2)
            self.win.move(1, self.input.cursor_x + 1)
        self.win.refresh()
        return

    def update_focus(self, focus: bool):
        super().update_focus(focus)
        self.update_cursor()
        return

    def update_content(self, content: str = ""):
        super().update_content(content=self.input.content, attr=self.confirmed * curses.color_pair(2))
        self.update_cursor()
        return

class SelectWindow(Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focus_line_index = 0
        self.start_char = 0
        self.selected = ''
        self.separator = ' '
        self.id = "selection"

    def update_content(self, content: str = None, separator: str = ' '):
        lines, start_cols = super().update_content(content)
        if self.focus:
            tmp_index = self.focus_line_index - self.start_line
            focus_line = lines[tmp_index]
            if separator:
                self.start_char = focus_line.rfind(separator) + len(separator)
            else:
                self.start_char = 0
            self.selected = focus_line[self.start_char:]
            self.win.addstr(self.focus_line_index - self.start_line + 1, start_cols[tmp_index] + self.start_char + 1, focus_line[self.start_char:], curses.color_pair(2))
            self.win.refresh()
        return

    def next(self):
        content_nlines = len(self.content.split('\n'))
        self.focus_line_index = min(self.focus_line_index + 1, content_nlines - 1)
        self.start_line = max(self.start_line, self.focus_line_index - self.nlines + 1)
        return

    def prev(self):
        self.focus_line_index = max(self.focus_line_index - 1, 0)
        self.start_line = min(self.focus_line_index, self.start_line)
        return

class EvaluationWindow(SelectWindow):
    def __init__(self, *args, content="completeness: 5\nclarity: 5\nrelevance: 5\ndepth_of_understanding: 5\ncoherence: 5\nexecute_time: 5\nadditional_notes: ", **kwargs):
        self.content = content
        self.lines = self.content.split('\n')
        self.input = Input()
        self.focus_line_index = 0
        self.start_index = 0
        self.selected = ''
        self.additional_notes = ""
        super().__init__(*args, content=content, **kwargs)
        return

    def update_content(self, content: str = None):
        if content is None:
            content = self.content
        else:
            self.content = content
        content += self.input.content
        self.win.clear()
        self.win.border()
        self.update_focus(self.focus)
        lines = content.split('\n')
        for i in range(min(self.nlines, len(lines) - self.start_line)):
            self.win.addstr(i + 1, 1, lines[i + self.start_line][:self.ncols])
        if self.focus == True:
            focus_line = lines[self.focus_line_index]
            self.start_index = focus_line.rfind(': ') + 2
            self.selected = focus_line[self.start_index:]
            self.win.addstr(self.focus_line_index - self.start_line + 1, self.start_index + 1, focus_line[self.start_index:], curses.color_pair(2))
        self.win.refresh()
        return

    def inc_value(self):
        int_value = int(self.selected)
        if int_value < 10:
            self.update_value(str(int_value + 1))
        return

    def dec_value(self):
        int_value = int(self.selected)
        if int_value > 0:
            self.update_value(str(int_value - 1))
        return

    def update_value(self, value):
        self.lines[self.focus_line_index] = self.lines[self.focus_line_index][:self.start_index] + str(value)
        self.update_content('\n'.join(self.lines) + self.input.content)
        return

    def export_evaluation_matrix(self):
        eval_pairs = [line.split(": ") for line in self.lines]
        for i in range(len(eval_pairs) - 1):
            eval_pairs[i][1] = int(eval_pairs[i][1])
        eval_pairs[-1][-1] = self.input.content
        matrix = {item: value for item, value in eval_pairs}
        return matrix

    def submit_evaluation(self, data_service, frames, settings, blending_result):
        data_service.create_result(
            frames=frames,
            settings=settings,
            blending_result=blending_result,
            evaluations=self.export_evaluation_matrix()
        )
        return

### Window Management Classes (Unchanged) ###

class WindowGroup:
    def __init__(self):
        self.focus_index = [0, 0]
        self.wins = [
            [],  # frame input windows
            [],  # hierarchy windows
            [],  # blending result windows
        ]
        return

    def add(self, win, column=0):
        self.wins[column].append(win)
        self.update_windows_focus()
        return

    def add_frame_input(self, start_y, start_x):
        title = f"Frame {len(self.wins[0])}"
        win_input = InputWindow(title, start_y, start_x, ncols=28)
        self.add(win_input)
        return

    def add_frame_hierarchy(self, relation, hierarchy):
        title = relation
        content = str(hierarchy).strip()
        win = SelectWindow(title, 0, self.wins[0][0].end_yx()[1], nlines=min(stdscr_height-2, len(content.split('\n'))), content=content)
        self.wins[1].append(win)
        return

    def add_blending_result(self, maxcols, text):
        if self.wins[1]:
            self.remove_frame_hierarchy()
        if self.wins[2]:
            self.remove_blending_result()
        win = Window("Blending Result", 0, self.wins[0][0].end_yx()[1], stdscr_height-7, maxcols, content=text)
        self.wins[2].append(win)
        return win

    def add_evaluation(self, maxcols):
        eval_win = EvaluationWindow("Press Tab to give feedback", stdscr_height-5, self.wins[0][0].end_yx()[1], 3, maxcols)
        self.wins[2].append(eval_win)

    def remove_frame_input(self):
        if len(self.wins[0]) > 2:
            self.wins[0][-1].win.clear()
            self.wins[0][-1].win.refresh()
            del self.wins[0][-1]
        return

    def remove_frame_hierarchy(self):
        win = self.wins[1][0]
        win.win.clear()
        win.win.refresh()
        del self.wins[1][0]
        return

    def remove_blending_result(self):
        for win in self.wins[2]:
            win.win.clear()
            win.win.refresh()
        self.wins[2] = []
        return

    def update_windows_focus(self):
        for col in self.wins:
            for win in col:
                win.update_focus(False)
        win = self.focus_win()
        win.update_focus(True)
        return

    def next_focus(self):
        self.focus_index[1] = (self.focus_index[1] + 1) % len(self.wins[self.focus_index[0]])
        self.update_windows_focus()
        return

    def prev_focus(self):
        self.focus_index[1] = (self.focus_index[1] - 1 + len(self.wins[0])) % len(self.wins[0])
        self.update_windows_focus()
        return

    def enter_focus(self, win_index: list):
        self._tmp_focus_index = self.focus_index
        self.focus_index = win_index
        self.update_windows_focus()
        return

    def quit_focus(self):
        win = self.focus_win()
        win.update_focus(False)
        win.update_content()
        self.focus_index = self._tmp_focus_index
        self.update_windows_focus()
        return

    def focus_win(self):
        win = self.get_win(self.focus_index)
        return win

    def get_win(self, index: list):
        col, i = index
        return self.wins[col][i]

class CycleList:
    def __init__(self, items: list):
        self.items = items
        self.index = 0
        return

    def now(self):
        return self.items[self.index]

    def next(self):
        self.index = (self.index + 1) % len(self.items)
        return

    def prev(self):
        self.index = (self.index - 1 + len(self.items)) % len(self.items)
        return

    def reset_index(self):
        self.index = 0
        return

class SettingsWindow(SelectWindow):
    def __init__(self, *args, **kwargs):
        self.settings = {
            "relations": CycleList([
                "Inheritance: children",
                "Inheritance: parents",
                "Perspective: children",
                "Perspective: parents",
                "Usage: children",
                "Usage: parents",
                "Subframe: children",
                "Subframe: parents"
            ]),
            "prompting": CycleList([
                "zero-shot",
                "one-shot",
                "few-shot",
                "chain-of-thought"
            ]),
            "rhetorical": CycleList([
                "rhetorical",
                "non-rhetorical"
            ])
        }
        super().__init__(*args, **kwargs)
        return

    def to_text(self) -> str:
        now_list = [item.now() for item in self.settings.values()]
        return '\n'.join(now_list)

    def update_content(self, *args, **kwargs):
        super().update_content(self.to_text(), separator=None)
        return

    def now_setting(self) -> CycleList:
        return list(self.settings.values())[self.focus_line_index]

    def export(self) -> list:
        res = [self.settings["prompting"].now(), self.settings["rhetorical"].now()]
        return res

### Main Function ###

def main(stdscr):
    global stdscr_height, stdscr_width

    # Initialize global variables
    stdscr_height, stdscr_width = stdscr.getmaxyx()
    langchain_service = LangChainService()

    stdscr.clear()
    stdscr.refresh()

    prompts = Prompts()

    # Logo Window
    win_logo_content = r"""
    ______                             ____  __               __         
   / ____/________ _____ ___  ___     / __ )/ /__  ____  ____/ /__  _____
  / /_  / ___/ __ `/ __ `__ \/ _ \   / __  / / _ \/ __ \/ __  / _ \/ ___/
 / __/ / /  / /_/ / / / / / /  __/  / /_/ / /  __/ / / / /_/ /  __/ /    
/_/   /_/   \__,_/_/ /_/ /_/\___/  /_____/_/\___/_/ /_/\__,_/\___/_/     
"""[1:]
    logo_lines = win_logo_content.split('\n')
    start_y = stdscr_height - len(logo_lines) - 2
    start_x = stdscr_width - max(len(line) for line in logo_lines) - 2
    win_logo = Window("", start_y, start_x, content=win_logo_content)

    # Key bindings window
    win_key_content = """
ESC:        Quit
+/-:        Add/Remove frame
Tab:        Switch window
Arrow
 up/down/   Change settings
 left/right:
Enter:      Enter Hierarchy
          & Confirm frame
/:          Start blending"""
    win_key = Window("Keys", 0, 0, content=win_key_content)

    # Window group and settings
    wg = WindowGroup()
    win_settings = SettingsWindow("Settings", 0, win_key.end_yx()[1], ncols=28, nlines=3, center=True)
    wg.add(win_settings, 0)

    # Frame hierarchy roots
    hierarchy_roots = {}
    frames = get_frames(FRAME_FOLDER)
    for item in win_settings.settings["relations"].items:
        relation, dir = item.split(": ")
        if dir == "children":
            hierarchy_roots[item] = analyze_hierarchy(frames, relation, encoding=ENCODING)
        else:
            hierarchy_roots[item] = analyze_hierarchy(frames, relation, reverse_order=True, encoding=ENCODING)

    wg.add_frame_input(win_settings.end_yx()[0], win_key.end_yx()[1])

    # Vector store loading window
    win_query_engine = Window("Query Engine", win_key.end_yx()[0], 0, ncols=win_key.ncols, content="Loading", center=True)
    loading_thread = threading.Thread(target=lambda: background_loading(win_query_engine, langchain_service))
    loading_thread.start()

    # Evaluation data service
    ds = EvaluationDataService(EVALUATION_CONFIG)

    while True:
        # Refresh windows
        if len(wg.wins) < 3 or not wg.wins[2]:
            win_logo.update_content()
        win_settings.update_content()
        wg.focus_win().update_content()

        key = stdscr.getch()

        # Quit
        if key == 27:  # ESC
            break

        # Focus on first column (input and settings)
        elif wg.focus_index[0] == 0:
            if key == ord('+'):
                wg.add_frame_input(wg.wins[0][-1].end_yx()[0], win_key.end_yx()[1])
            elif key == ord('-'):
                wg.remove_frame_input()
            elif key == ord('\t') or key == 9:  # TAB
                wg.next_focus()
            else:
                if wg.focus_win().id == "input":
                    if key == ord('\n'):
                        if wg.focus_win().content and wg.wins[1]:
                            wg.enter_focus([1, 0])
                            continue
                    # Generate frame blending example with LangChain
                    elif key == ord("/"):
                        if langchain_service.vector_store is None:
                            response = "Vector store is not ready!"
                        else:
                            confirmed_frames = [win.content for win in wg.wins[0] if win.id == "input" and win.confirmed]
                            if not confirmed_frames:
                                response = "No frames confirmed!"
                            else:
                                prompting_strategy = win_settings.settings["prompting"].now()
                                rhetorical = win_settings.settings["rhetorical"].now() == "rhetorical"
                                instructions = prompts[prompting_strategy](rhetorical)
                                frame_blending_template = f"""
{instructions}

Below is some context about frames (extracted from the user’s frame collection):
{{context}}

The user wants to blend the following frames: {{input}}

Please produce:
1. A short example sentence or expression that illustrates how these frames blend.
2. A concise analysis explaining the input spaces, cross-space mapping, blended space, and emergent structure.

If you are missing crucial info or are unsure, respond with "I don't know."

Answer concisely, in a professional style.
"""
                                prompt_template = PromptTemplate(
                                    template=frame_blending_template,
                                    input_variables=["context", "input"]
                                )
                                rag_chain = langchain_service.build_rag_chain(prompt_template)
                                frames_str = ", ".join(confirmed_frames)
                                response = langchain_service.generate_response(rag_chain, frames_str)
                        maxcols = stdscr_width - wg.wins[0][0].end_yx()[1] - 2
                        wg.add_blending_result(maxcols=maxcols, text=response)
                        if langchain_service.vector_store is not None or DEBUG:
                            wg.add_evaluation(maxcols=maxcols)
                        wg.enter_focus([2, 0])
                        continue
                    elif wg.focus_win().confirmed == True:
                        if key == curses.KEY_BACKSPACE or key == 127:
                            wg.focus_win().confirmed = False
                    else:
                        win = wg.focus_win()
                        win.input.execute(key)
                        win.update_content(win.input.content)
                elif wg.focus_win().id == "selection":
                    if key == curses.KEY_DOWN:
                        wg.focus_win().next()
                    elif key == curses.KEY_UP:
                        wg.focus_win().prev()
                    elif key == curses.KEY_RIGHT:
                        wg.focus_win().now_setting().next()
                    elif key == curses.KEY_LEFT:
                        wg.focus_win().now_setting().prev()

            # Update frame_hierarchy window
            if wg.wins[1]:
                wg.remove_frame_hierarchy()
            win = wg.focus_win()
            if win.id == "input" and not win.confirmed:
                relation = win_settings.settings["relations"].now()
                root = hierarchy_roots[relation]
                hierarchy = root.find(win.content)
                if hierarchy:
                    wg.add_frame_hierarchy(relation, hierarchy)

        # Focus on second column (hierarchy)
        elif wg.focus_index[0] == 1:
            if key == ord('\\'):
                wg.quit_focus()
            elif key == curses.KEY_DOWN:
                wg.wins[1][0].next()
            elif key == curses.KEY_UP:
                wg.wins[1][0].prev()
            elif key == curses.KEY_RIGHT:
                win_settings.settings["relations"].next()
            elif key == curses.KEY_LEFT:
                win_settings.settings["relations"].prev()
            elif key == ord('\n'):
                frame = wg.wins[1][0].selected
                wg.quit_focus()
                wg.remove_frame_hierarchy()
                wg.focus_win().input.content = frame
                wg.focus_win().update_content(frame)
                wg.focus_win().confirmed = True
                continue
            if wg.wins[1][0].title != win_settings.settings["relations"].now():
                wg.remove_frame_hierarchy()
                relation = win_settings.settings["relations"].now()
                root = hierarchy_roots[relation]
                hierarchy = root.find(wg.get_win(wg._tmp_focus_index).content)
                wg.add_frame_hierarchy(relation, hierarchy)
                wg.wins[1][0].update_focus(True)

        # Focus on third column (results and evaluation)
        elif wg.focus_index[0] == 2:
            win = wg.focus_win()
            if key == ord('\t') or key == 9:  # TAB
                wg.next_focus()
            elif key == ord('\\'):
                wg.quit_focus()
                wg.remove_blending_result()
            elif wg.focus_index[1] == 0:
                if key == curses.KEY_DOWN:
                    win.scroll_down()
                elif key == curses.KEY_UP:
                    win.scroll_up()
            elif wg.focus_index[1] == 1:
                if key == ord('\n'):
                    win.submit_evaluation(
                        data_service=ds,
                        frames=[win.content for win in wg.wins[0] if win.id == "input" and win.confirmed],
                        settings=win_settings.export(),
                        blending_result=wg.wins[2][0].content
                    )
                    wg.quit_focus()
                    wg.remove_blending_result()
                elif key == curses.KEY_DOWN:
                    win.next()
                elif key == curses.KEY_UP:
                    win.prev()
                elif win.focus_line_index < len(win.lines) - 1:
                    if key == curses.KEY_LEFT:
                        win.dec_value()
                    elif key == curses.KEY_RIGHT:
                        win.inc_value()
                else:
                    win.input.execute(key)
                    win.update_content()

    return

### Background Loading Function ###

def background_loading(window, langchain_service):
    """
    Runs load_packages in a background thread and updates the loading window.
    """
    text = langchain_service.load_packages()
    if text == "Finished":
        window.update_content("Vector store loaded", curses.color_pair(3))
    else:
        window.update_content(text, curses.color_pair(4))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding', type=str, default='utf-8', help='Specify the encoding')
    parser.add_argument('--debug', type=bool, default=False, help="Debug mode when not running on GPU")
    args = parser.parse_args()
    ENCODING = args.encoding
    DEBUG = args.debug

    curses.initscr()
    curses.start_color()
    curses.init_pair(1, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    try:
        curses.wrapper(main)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr