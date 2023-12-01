import json
import csv

class JSONToCSVConverter:
    def __init__(self, json_file_path, csv_file_path, headers):
        self.json_file_path = json_file_path
        self.csv_file_path = csv_file_path
        self.headers = headers

    def convert(self):
        # Load JSON data from the file
        with open(self.json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Extract data from JSON and write to CSV
        rows = self.extract_data(data)
        self.write_to_csv(rows)

    def extract_data(self, data):
        rows = []

        for item in data:
            title = item["title"]
            post = item["post"]
            url = item["url"]
            source = item["source"]
            timestamp = item["timestamp"]

            for comment in item["comments"]:
                text = comment["text"]
                user_name = comment["user_name"]
                user_id = comment["user_id"]
                comment_url = comment["url"]
                comment_timestamp = comment["timestamp"]

                for annotation in comment["annotations"]:
                    annotation_type = annotation["type"]
                    annotation_target = annotation["target"]
                    annotation_annotator = annotation["annotator"]

                    rows.append([title, post, url, source, timestamp, text, user_name, user_id, comment_url,
                                 comment_timestamp, annotation_type, annotation_target, annotation_annotator])

        return rows

    def write_to_csv(self, rows):
        # Write the CSV file
        with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(self.headers)
            csv_writer.writerows(rows)

# Example usage
json_to_csv_converter = JSONToCSVConverter('/data/ARENAS_Automatic_Extremist_Analysis/ARENAS_Automatic_Extremist_Analysis/Data/FRENK/migrants-en.json',
                                           '/data/ARENAS_Automatic_Extremist_Analysis/ARENAS_Automatic_Extremist_Analysis/Data/FRENK/migrants-en.csv',
                                           ["title", "post", "url", "source", "timestamp", "text", "user_name",
                                            "user_id", "comment_url", "comment_timestamp", "annotation_type",
                                            "annotation_target", "annotation_annotator"])

json_to_csv_converter.convert()

