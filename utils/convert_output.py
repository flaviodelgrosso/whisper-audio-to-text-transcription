class Formatter:
    @classmethod
    def preamble(cls):
        return "TRANSCRIPTION\n\n"

    @classmethod
    def format_seconds(cls, seconds):
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk['text']
        start, end = chunk['timestamp'][0], chunk['timestamp'][1]
        start_format, end_format = cls.format_seconds(
            start), cls.format_seconds(end)
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"


def convert(data, output_path):
    string = Formatter.preamble()
    for index, chunk in enumerate(data['chunks'], 1):
        entry = Formatter.format_chunk(chunk, index)
        string += entry

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(string)
