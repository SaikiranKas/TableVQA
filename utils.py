from bs4 import BeautifulSoup

def remove_all_attributes(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
    for tag in soup.find_all(True):
        tag.attrs = {}
    return str(soup)

def clean_html(html_):
    def replace_html_attr(html_):
        tag_list = ["<thead>", "</thead>", "<tbody>", "</tbody>", "<sup>", "</sup>", "<sub>", "</sub>", "\xa0", "<p>", "</p>"]
        tag_list += [f'colspan="{i}"' for i in range(31)]
        tag_list += [f'rowspan="{i}"' for i in range(31)]
        for tag in tag_list:
            html_ = html_.replace(tag, "")
        html_ = html_.replace("<th", "<td").replace("</th>", "</td>")
        return html_
    
    html_ = ' '.join(html_.split())
    return replace_html_attr(html_)

def preprocess(html):
    return clean_html(remove_all_attributes(html))
