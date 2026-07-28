Optera 24-Hour Challenge — Starter Image Set (47 images)
========================================================

Every image here is REAL Optera data: phone photos sent in over WhatsApp by two
different transport clients, exactly as our pipeline receives them in production.
This is deliberately a HETEROGENEOUS stream — the whole point is that a real
inbox is not one clean document type. You'll find, mixed together and unlabelled:

  - Handwritten mechanic "work reports" — per-day logs, ruled forms and plain
    notebook pages, keyed by bus/vehicle code (e.g. TCM35, MAC-1), a date, the
    work done, mechanic name, material column. English + Hindi + Gujarati,
    strikethroughs, multi-colour ink, fingers over the page. (client A)
  - Printed vendor bills / invoices — tyre, parts, greasing, seat-cover shops
    (Anupam, Tiwari Auto Parts, Sindhi Aslam Iqbal, Sajid Tyre Service, ...),
    with printed headers and handwritten line items, amounts, GST, vehicle no. (client B)
  - Meter / dashboard photos — digital odometer clusters (e.g. 320654 km) and
    AdBlue/DEF dispenser readings (Rs / Litres / Rs-per-litre). (client B)
  - Object photos that are NOT documents at all — a battery on the ground, a bare
    tyre, a number plate. These carry a little text but no extractable record. (client B)

That last category matters. Part of the job is routing: decide what each image
IS, extract the ones that are structured records, and REFUSE the ones that
aren't — a photo of a battery must not become a neat invoice row. A pipeline that
hallucinates structure out of a battery photo is worse than one that abstains.

What's yours to define: the canonical schema(s) these documents map to, and how
you classify / route / reject. Filenames (optera_doc_NN.jpg) are intentionally
unlabelled — you don't get told which is which.

This is a STARTER set, not the whole test. You may add more of your own to harden
it, but we will evaluate your committed pipeline on other, unseen Optera images,
so don't overfit to these 47.

Ordinary operational data only — please use it just for this challenge and don't
redistribute it.
