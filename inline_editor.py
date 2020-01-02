from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import (
    HSplit,
    Window,
)
from prompt_toolkit.styles import Style
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import (
    TextArea,
)


def edit_multiline(default_text=""):
    kb = KeyBindings()

    @kb.add('c-q')
    @kb.add('escape', 'enter')
    def exit_(event):
        """
        Pressing Ctrl-Q, Alt+Enter or Esc + Enter will exit the editor.
        """
        bottom_bar.style="hidden"
        event.app.exit(text_field.text)

    @kb.add('c-c')
    def do_copy(event):
        data = text_field.buffer.copy_selection()
        get_app().clipboard.set_data(data)

    @kb.add('c-x', eager=True)
    def do_cut(event):
        data = text_field.buffer.cut_selection()
        get_app().clipboard.set_data(data)

    @kb.add('c-z')
    def do_undo(event):
        text_field.buffer.undo()

    @kb.add('c-y')
    def do_redo(event):
        text_field.buffer.redo()

    @kb.add('c-a')
    def do_select_all(event):
        text_field.buffer.cursor_position = 0
        text_field.buffer.start_selection()
        text_field.buffer.cursor_position = len(text_field.buffer.text)

    @kb.add('c-v')
    def do_paste(event):
        text_field.buffer.paste_clipboard_data(get_app().clipboard.get_data())

    @kb.add('left')
    def leftarrow(event):
        text_field.buffer.selection_state = None
        if text_field.buffer.cursor_position != 0 and text_field.text[text_field.buffer.cursor_position-1] == '\n':
            text_field.buffer.cursor_up()
            text_field.buffer.cursor_right(len(text_field.text))
        else:
            text_field.buffer.cursor_left()

    @kb.add('right')
    def rightarrow(event):
        text_field.buffer.selection_state = None
        if text_field.buffer.cursor_position < len(text_field.text) and text_field.text[text_field.buffer.cursor_position] == '\n':
            text_field.buffer.cursor_down()
            text_field.buffer.cursor_left(len(text_field.text))

        else:
            text_field.buffer.cursor_right()

    @kb.add('up')
    def uparrow(event):
        text_field.buffer.selection_state = None
        text_field.buffer.cursor_up()

    @kb.add('down')
    def downarrow(event):
        text_field.buffer.selection_state = None
        text_field.buffer.cursor_down()


    text_field = TextArea()
    bottom_bar=Window(content=FormattedTextControl(text='\nCurrently editing. Press Ctrl+Q, Alt+Enter or Esc + Enter to exit.'))

    root_container = HSplit([
        text_field,
        bottom_bar,
    ])

    layout = Layout(root_container)

    app = Application(key_bindings=kb, layout=layout, enable_page_navigation_bindings=True, full_screen=False)
    text_field.text=default_text
    text_field.buffer.cursor_position = len(text_field.buffer.text)
    text = app.run()

    return text


if __name__ == "__main__":
    edit_multiline("Testing\nwith\na\ndefault\nstring.")
