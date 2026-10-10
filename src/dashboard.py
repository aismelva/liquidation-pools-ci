    book = bids if side_book == "bid" else asks
    return sum(px * qty for px, qty in book.items()) or float("nan")
