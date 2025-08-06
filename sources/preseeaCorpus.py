from sources.common.common import processControl, logger, log_
from sources.common.utils import save_results, get_next_combination

from playwright.sync_api import sync_playwright
import time
import re
from os.path import join
import asyncio
import sys
import os
from urllib.parse import urljoin
from tqdm import tqdm

import aiohttp
from playwright.async_api import async_playwright, TimeoutError

def preseeaAccess():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        url = processControl.defaults['preseeaUrl']
        # Ir a PRESEEA
        page.goto(url)

        # Esperar a que cargue
        page.wait_for_selector("#busca2")  # Asegúrate de que sea el ID correcto

        # Ingresar la consulta
        consulta = "[(lemma='iglesia'%c)] :: match.text_sexo = 'H' | match.text_sexo = 'M'"
        page.fill("#busca2", consulta)

        # Hacer clic en el botón
        page.click("#sencillas > fieldset > input[type=button]:nth-child(4)")

        # Esperar resultados
        page.wait_for_selector("#superResultados > fieldset")  # Ajustar según estructura real

        # Extraer resultados
        print(page.inner_text("#superResultados > fieldset"))  # o `.result-count`

        # (Opcional) Descargar CSV si está disponible

        browser.close()



async def capture_document_content(page, retries=3):
    """
    @Desc: Captures the content of a webpage with retry logic in case of failures.
    @Params: page (Page) - Page object from Playwright session, retries (int) - Maximum number of retry attempts.
    @Returns: Page content as a string if successful.
    @Raises: TimeoutError if content capture fails after specified retries.
    """
    for attempt in range(retries):
        try:
            await page.wait_for_load_state("networkidle")
            return await page.content()
        except Exception as error:
            log_("warning", logger, f"Attempt {attempt + 1} failed: {error}")
            await asyncio.sleep(1)
    raise TimeoutError(f"Failed to capture content after {retries} attempts. URL: {processControl.defaults['preseeaUrl']}")


async def capture_document(playwright):
    """
    @Desc: Launches a headless browser session to capture HTML content from a specified URL.
    @Params: playwright (Playwright) - Playwright session instance.
    @Returns: Cleaned HTML content as a string.
    @Raises: ValueError if page loading or content capture encounters an error.
    """
    log_("debug", logger, f"Starting download: {processControl.defaults['preseeaUrl']}")
    start_time = time.time()

    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    page = await context.new_page()
    processControl.stage = "crawl"


    page_timeout =  30000

    try:
        response = await page.goto(processControl.defaults['preseeaUrl'], timeout=page_timeout)
        load_duration = time.time() - start_time
        log_("debug", logger, f"Downloaded in {load_duration:.1f}s, URL: {page.url}")

        headers = response.headers
        last_modified = headers.get('last-modified')
        if last_modified:
            log_("debug", logger, f"Headers: Last-Modified: {last_modified}")

        page.wait_for_selector("input#busca2", state="visible")
        # Espera adicional para asegurar que el campo esté listo
        time.sleep(3)  # ⏳ Ajusta si hace falta

        # Usa type() para simular escritura real
        campo = page.locator("input#busca2")
        campo.click()
        time.sleep(2)
        campo.type("[(lemma='iglesia'%c)] :: match.text_sexo = 'H' | match.text_sexo = 'M'", delay=30)

        page.locator("input#busca2").fill("[(lemma='iglesia'%c)] :: match.text_sexo = 'H' | match.text_sexo = 'M'")

        # Ingresar la consulta
        consulta = "[(lemma='iglesia'%c)] :: match.text_sexo = 'H' | match.text_sexo = 'M'"
        page.evaluate("""
            () => {
                const input = document.querySelector("#busca2");
                input.value = "[(lemma='iglesia'%c)] :: match.text_sexo = 'H' | match.text_sexo = 'M'";
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        """)

        page.locator("#busca").type("[(lemma='iglesia'%c)] :: match.text_sexo = 'H' | match.text_sexo = 'M'")

        # Hacer clic en el botón
        page.click("#sencillas > fieldset > input[type=button]:nth-child(4)")

        # Esperar resultados
        page.wait_for_selector("#superResultados > fieldset")  # Ajustar según estructura real

        # Extraer resultados
        print(page.inner_text("#superResultados > fieldset"))  # o `.result-count`



        content = await capture_document_content(page)
    except Exception as error:
        raise ValueError(f"Error loading page: {error}") from error
    finally:
        await page.close()
        await browser.close()

    # Remove whitespace between tags for cleaner HTML
    content = re.sub(r'>\s+<', '><', content)

    return content


async def process_document_url(session):
    """
    @Desc: Manages the document processing workflow, including HTML extraction, language validation, and file saving.
    @Params: session (ClientSession) - aiohttp session for HTTP requests.
    @Returns: None
    @Raises: ValueError if text extraction, language validation, or control updates fail.
    """
    start_time = time.time()
    log_("info", logger, f"Starting new download task for {processControl.defaults['preseeaUrl']}")
    try:
        async with async_playwright() as playwright:
            html_content = await capture_document(playwright)

    except Exception as error:
        log_("warning", logger, f"Error processing document: {error}")



async def process_apps():
    """
    @Desc: Initializes and manages an aiohttp session to process all application controls.
    @Returns: Status and message summary of the processing operation.
    """
    timeout = aiohttp.ClientTimeout(total=200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await process_document_url(session)


async def download_audio_files(audio_links, identificador):
    output_dir = os.path.join(processControl.env['outputDir'], identificador)
    os.makedirs(output_dir, exist_ok=True)

    # Download each audio file
    base_url = "https://preseea.uah.es/corpus/"
    base_url = processControl.defaults['baseDownUrl']
    downloaded_files = []
    count = 0

    async with aiohttp.ClientSession() as session:
        with tqdm(total=len(audio_links), unit='file', desc='Downloading') as pbar:
            for link in audio_links:
                try:
                    # Construct full URL
                    count += 1
                    full_url = urljoin(base_url, link)
                    filename = os.path.basename(link)
                    output_path = os.path.join(output_dir, f"{count}_{filename}")

                    # Download the file using aiohttp
                    async with session.get(full_url) as response:
                        if response.status == 200:
                            total_size = int(response.headers.get('content-length', 0))
                            with open(output_path, 'wb') as f, tqdm(
                                total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                                miniters=1, desc=filename, leave=False
                            ) as file_pbar:
                                async for chunk in response.content.iter_chunked(1024):
                                    f.write(chunk)
                                    file_pbar.update(len(chunk))
                            downloaded_files.append(output_path)
                        else:
                            raise Exception(f"Failed to download {full_url}, status: {response.status}")

                    pbar.set_postfix(file=filename[-15:])  # Show last 15 chars of filename
                    pbar.update(1)

                except Exception as e:
                    print(f"\nFailed to download {link}: {str(e)}")
                    pbar.update(1)  # Still advance main progress bar on failure

    return downloaded_files

async def mainProcessCrawler(lemma, sexo, edad, estudios):
    async with async_playwright() as p:
        identificador = f"{lemma}-{sexo}-{edad}"
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        #await page.goto("https://preseea.uah.es/corpus-preseea")
        await page.goto(processControl.defaults['preseeaUrl'])
        await page.wait_for_selector("iframe.wrapped-iframe")

        # Obtener el iframe
        iframe_element = await page.wait_for_selector("iframe.wrapped-iframe")
        iframe = await iframe_element.content_frame()

        # Esperar el campo de búsqueda dentro del iframe
        await iframe.wait_for_selector("input#busca2", state="visible")
        campo = iframe.locator("input#busca2")
        await campo.click()
        #await campo.type(' [(word="bar")] :: (match.text_ciudad = "Alcalá de Henares" | match.text_ciudad = "Barcelona" | match.text_ciudad = "Cádiz" | match.text_ciudad = "Gijón" | match.text_ciudad = "Granada" | match.text_ciudad = "Las Palmas de Gran Canaria" | match.text_ciudad = "Madrid" | match.text_ciudad = "Málaga" | match.text_ciudad = "Palma de Mallorca" | match.text_ciudad = "Santander" | match.text_ciudad = "Santiago de Compostela" | match.text_ciudad = "Sevilla" | match.text_ciudad = "Valencia") & (match.text_sexo = "H")   & (match.text_edad = "1") & (match.text_nivel = "1") &  match.turno_inf = "I.*" within turno', delay=30)

        await campo.type(
            f' [(word="{lemma}")] :: (match.text_ciudad = "Alcalá de Henares" | '
            'match.text_ciudad = "Barcelona" | match.text_ciudad = "Cádiz" | '
            'match.text_ciudad = "Gijón" | match.text_ciudad = "Granada" | '
            'match.text_ciudad = "Las Palmas de Gran Canaria" | match.text_ciudad = "Madrid" | '
            'match.text_ciudad = "Málaga" | match.text_ciudad = "Palma de Mallorca" | '
            'match.text_ciudad = "Santander" | match.text_ciudad = "Santiago de Compostela" | '
            'match.text_ciudad = "Sevilla" | match.text_ciudad = "Valencia") '
            f'& (match.text_sexo = "{sexo}") & (match.text_edad = "{edad}") '
            #f'& (match.text_nivel = "{estudios}") & match.turno_inf = "I.*" within turno',
            f'& match.turno_inf = "I.*" within turno',
            delay=30
        )
        # Hacer clic en el botón para crear consulta (también dentro del iframe)
        await iframe.click("#sencillas > fieldset > input[type=button]:nth-child(4)")

        await iframe.wait_for_selector("#superResultados", timeout=10000)

        # Extraer el HTML del resultado
        resultados_html = await iframe.locator("#superResultados .consultaframe").inner_html()

        # Extraer los enlaces de audio
        audio_links = await iframe.locator("#superResultados a[href^='audio/'][download]").evaluate_all(
            """elements => elements.map(el => el.href)"""
        )

        # Cerrar el browser antes de descargar
        await browser.close()

        # Descargar los archivos de audio
        downloaded = await download_audio_files(audio_links, identificador)

        # Buscar los números con regex
        match = re.search(r"N.º total de ejemplos:\s*<b>(\d+)</b>\s*en\s*<b>(\d+)</b>", resultados_html)


        if match:
            nn = int(match.group(1))  # Ejemplos
            mm = int(match.group(2))  # Entrevistas
        else:
            nn = 0
            mm = 0
            log_("error", logger, resultados_html)

        log_("info", logger, f"Ejemplos encontrados: {nn}")
        log_("info", logger, f"Entrevistas: {mm}")

        return {
            "Ejemplos": nn,
            "Entrevistas": mm,
            "downloadAudio": downloaded
        }

def processCrawler(resultsPath):
    """
    @Desc: Entry-point function to run the asynchronous process for document capture and processing.
    @Returns: None
    """

    next_comb = get_next_combination(resultsPath)

    if next_comb is None:
        log_("info", logger, "Se han utilizado todas las combinaciones")
        return None

    estudios = "1"
    lemma = next_comb["lemma"]
    sexo = next_comb["sexo"]
    edad = next_comb["edad"]

    results = asyncio.run(mainProcessCrawler(lemma, sexo, edad, estudios))
    save_results(resultsPath, lemma, sexo, edad, results)
    log_("info", logger, f"Processed combination: {lemma}, {sexo}, {edad}. Results saved to {resultsPath}")
