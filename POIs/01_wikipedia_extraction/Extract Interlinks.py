import wikipedia, json, pathlib

wikipedia.set_lang("he")
def get_interlink_titles(title):
    try:
        wiki_page = wikipedia.page(title)
        interlink_titles = wiki_page.links
        two_word_titles = [t for t in interlink_titles if len(t.split()) >= 1]
        out = pathlib.Path(f"{title}_titles.json")
        out.write_text(json.dumps(two_word_titles, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved to {out.resolve()}")

    except wikipedia.exceptions.PageError:
        print(f"Error: Page '{title}' not found.")
    except wikipedia.exceptions.RedirectError as e:
        print(f"Error with redirect: {e}")

pages = ["הרפורמה המשפטית בישראל (2023)", "התמיכה ברפורמה המשפטית", "המחאה נגד הרפורמה המשפטית"]

for page in pages:
    get_interlink_titles(page)


